from __future__ import annotations
import concurrent.futures
import dataclasses
import functools
import inspect
import logging
import uuid
from datetime import datetime, timezone
from typing import (
from langchain_core._api import warn_deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables import config as runnable_config
from langchain_core.runnables import utils as runnable_utils
from langchain_core.tracers.evaluation import (
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.client import Client
from langsmith.env import get_git_info, get_langchain_env_var_metadata
from langsmith.evaluation import (
from langsmith.evaluation import (
from langsmith.run_helpers import as_runnable, is_traceable_function
from langsmith.schemas import Dataset, DataType, Example, Run, TracerSession
from langsmith.utils import LangSmithError
from requests import HTTPError
from typing_extensions import TypedDict
from langchain.callbacks.manager import Callbacks
from langchain.chains.base import Chain
from langchain.evaluation.loading import load_evaluator
from langchain.evaluation.schema import (
from langchain.smith import evaluation as smith_eval
from langchain.smith.evaluation import config as smith_eval_config
from langchain.smith.evaluation import name_generation, progress
@dataclasses.dataclass
class _DatasetRunContainer:
    """A container to help manage the state of a eval run."""
    client: Client
    project: TracerSession
    wrapped_model: MCF
    examples: List[Example]
    configs: List[RunnableConfig]
    batch_evaluators: Optional[List[smith_eval_config.BATCH_EVALUATOR_LIKE]] = None

    def _merge_test_outputs(self, batch_results: list, all_eval_results: Dict[str, _RowResult]) -> dict:
        results: dict = {}
        for example, output in zip(self.examples, batch_results):
            row_result = cast(_RowResult, all_eval_results.get(str(example.id), {}))
            results[str(example.id)] = {'input': example.inputs, 'feedback': row_result.get('feedback', []), 'execution_time': row_result.get('execution_time'), 'run_id': row_result.get('run_id')}
            if isinstance(output, EvalError):
                results[str(example.id)]['Error'] = output.Error
            else:
                results[str(example.id)]['output'] = output
            if example.outputs:
                results[str(example.id)]['reference'] = example.outputs
        return results

    def _run_batch_evaluators(self, runs: Dict[str, Run]) -> List[dict]:
        evaluators = self.batch_evaluators
        if not evaluators:
            return []
        runs_list = [runs[str(example.id)] for example in self.examples]
        aggregate_feedback = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for evaluator in evaluators:
                try:
                    result = evaluator(runs_list, self.examples)
                    if isinstance(result, EvaluationResult):
                        result = result.dict()
                    aggregate_feedback.append(cast(dict, result))
                    executor.submit(self.client.create_feedback, **result, run_id=None, project_id=self.project.id)
                except Exception as e:
                    logger.error(f'Error running batch evaluator {repr(evaluator)}: {e}')
        return aggregate_feedback

    def _collect_metrics(self) -> Tuple[Dict[str, _RowResult], Dict[str, Run]]:
        all_eval_results: dict = {}
        all_runs: dict = {}
        for c in self.configs:
            for callback in cast(list, c['callbacks']):
                if isinstance(callback, EvaluatorCallbackHandler):
                    eval_results = callback.logged_eval_results
                    for (_, example_id), v in eval_results.items():
                        all_eval_results.setdefault(str(example_id), {}).update({'feedback': v})
                elif isinstance(callback, LangChainTracer):
                    run = callback.latest_run
                    execution_time = (run.end_time - run.start_time).total_seconds() if run and run.end_time else None
                    run_id = str(run.id) if run else None
                    all_eval_results.setdefault(str(callback.example_id), {}).update({'execution_time': execution_time, 'run_id': run_id, 'run': run})
                    all_runs[str(callback.example_id)] = run
        return (cast(Dict[str, _RowResult], all_eval_results), all_runs)

    def _collect_test_results(self, batch_results: List[Union[dict, str, LLMResult, ChatResult]]) -> TestResult:
        logger.info('Waiting for evaluators to complete.')
        wait_for_all_evaluators()
        all_eval_results, all_runs = self._collect_metrics()
        aggregate_feedback = None
        if self.batch_evaluators:
            logger.info('Running session evaluators.')
            aggregate_feedback = self._run_batch_evaluators(all_runs)
        results = self._merge_test_outputs(batch_results, all_eval_results)
        return TestResult(project_name=self.project.name, results=results, aggregate_metrics=aggregate_feedback)

    def finish(self, batch_results: list, verbose: bool=False) -> TestResult:
        results = self._collect_test_results(batch_results)
        if verbose:
            try:
                agg_feedback = results.get_aggregate_feedback()
                _display_aggregate_results(agg_feedback)
            except Exception as e:
                logger.debug(f'Failed to print aggregate feedback: {repr(e)}')
        try:
            self.client.update_project(self.project.id, end_time=datetime.now(timezone.utc))
        except Exception as e:
            logger.debug(f'Failed to close project: {repr(e)}')
        return results

    @classmethod
    def prepare(cls, client: Client, dataset_name: str, llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY, project_name: Optional[str], evaluation: Optional[smith_eval.RunEvalConfig]=None, tags: Optional[List[str]]=None, input_mapper: Optional[Callable[[Dict], Any]]=None, concurrency_level: int=5, project_metadata: Optional[Dict[str, Any]]=None, revision_id: Optional[str]=None, dataset_version: Optional[Union[datetime, str]]=None) -> _DatasetRunContainer:
        project_name = project_name or name_generation.random_name()
        if revision_id:
            if not project_metadata:
                project_metadata = {}
            project_metadata.update({'revision_id': revision_id})
        wrapped_model, project, dataset, examples = _prepare_eval_run(client, dataset_name, llm_or_chain_factory, project_name, project_metadata=project_metadata, tags=tags, dataset_version=dataset_version)
        tags = tags or []
        for k, v in (project.metadata.get('git') or {}).items():
            tags.append(f'git:{k}={v}')
        run_metadata = {'dataset_version': project.metadata['dataset_version']}
        if revision_id:
            run_metadata['revision_id'] = revision_id
        wrapped_model = _wrap_in_chain_factory(llm_or_chain_factory)
        run_evaluators = _setup_evaluation(wrapped_model, examples, evaluation, dataset.data_type or DataType.kv)
        _validate_example_inputs(examples[0], wrapped_model, input_mapper)
        progress_bar = progress.ProgressBarCallback(len(examples))
        configs = [RunnableConfig(callbacks=[LangChainTracer(project_name=project.name, client=client, example_id=example.id), EvaluatorCallbackHandler(evaluators=run_evaluators or [], client=client, example_id=example.id, max_concurrency=0), progress_bar], tags=tags, max_concurrency=concurrency_level, metadata=run_metadata) for example in examples]
        return cls(client=client, project=project, wrapped_model=wrapped_model, examples=examples, configs=configs, batch_evaluators=evaluation.batch_evaluators if evaluation else None)