from __future__ import annotations
import collections
import concurrent.futures as cf
import datetime
import functools
import itertools
import logging
import pathlib
import threading
import uuid
from contextvars import copy_context
from typing import (
from requests import HTTPError
from typing_extensions import TypedDict
import langsmith
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import run_trees, schemas
from langsmith import utils as ls_utils
from langsmith.evaluation.evaluator import (
from langsmith.evaluation.integrations import LangChainStringEvaluator
class _ExperimentManager(_ExperimentManagerMixin):
    """Manage the execution of experiments.

    Supports lazily running predictions and evaluations in parallel to facilitate
    result streaming and early debugging.

    Args:
        data (DATA_T): The data used for the experiment. Can be a dataset name or ID OR
            a generator of examples.
        runs (Optional[Iterable[schemas.Run]]): The runs associated with the experiment
            predictions.
        experiment (Optional[schemas.TracerSession]): The tracer session
            associated with the experiment.
        experiment_prefix (Optional[str]): The prefix for the experiment name.
        metadata (Optional[dict]): Additional metadata for the experiment.
        client (Optional[langsmith.Client]): The Langsmith client used for
             the experiment.
        evaluation_results (Optional[Iterable[EvaluationResults]]): The evaluation
            sresults for the experiment.
        summary_results (Optional[Iterable[EvaluationResults]]): The aggregate results
            for the experiment.
    """

    def __init__(self, data: DATA_T, /, experiment: Optional[Union[schemas.TracerSession, str]], metadata: Optional[dict]=None, client: Optional[langsmith.Client]=None, runs: Optional[Iterable[schemas.Run]]=None, evaluation_results: Optional[Iterable[EvaluationResults]]=None, summary_results: Optional[Iterable[EvaluationResults]]=None):
        super().__init__(experiment=experiment, metadata=metadata, client=client)
        self._data = data
        self._examples: Optional[Iterable[schemas.Example]] = None
        self._runs = runs
        self._evaluation_results = evaluation_results
        self._summary_results = summary_results

    @property
    def examples(self) -> Iterable[schemas.Example]:
        if self._examples is None:
            self._examples = _resolve_data(self._data, client=self.client)
        self._examples, examples_iter = itertools.tee(self._examples)
        return examples_iter

    @property
    def dataset_id(self) -> str:
        if self._experiment is None or not getattr(self._experiment, 'reference_dataset_id', None):
            example = next(iter(self.examples))
            return str(example.dataset_id)
        return str(cast(schemas.TracerSessionResult, self._experiment).reference_dataset_id)

    @property
    def evaluation_results(self) -> Iterable[EvaluationResults]:
        if self._evaluation_results is None:
            return [{'results': []} for _ in self.examples]
        return self._evaluation_results

    @property
    def runs(self) -> Iterable[schemas.Run]:
        if self._runs is None:
            raise ValueError('Runs not provided in this experiment. Please predict first.')
        self._runs, runs_iter = itertools.tee(self._runs)
        return runs_iter

    def start(self) -> _ExperimentManager:
        first_example = next(itertools.islice(self.examples, 1))
        project = self._get_project(first_example)
        self._print_experiment_start(project, first_example)
        return self.__class__(self.examples, experiment=project, metadata=self._metadata, client=self.client, runs=self._runs, evaluation_results=self._evaluation_results)

    def with_predictions(self, target: TARGET_T, /, max_concurrency: Optional[int]=None) -> _ExperimentManager:
        """Lazily apply the target function to the experiment."""
        context = copy_context()
        _experiment_results = context.run(self._predict, target, max_concurrency=max_concurrency)
        r1, r2 = itertools.tee(_experiment_results, 2)
        return _ExperimentManager((pred['example'] for pred in r1), experiment=self._experiment, metadata=self._metadata, client=self.client, runs=(pred['run'] for pred in r2))

    def with_evaluators(self, evaluators: Sequence[Union[EVALUATOR_T, RunEvaluator]], *, max_concurrency: Optional[int]=None) -> _ExperimentManager:
        """Lazily apply the provided evaluators to the experiment."""
        evaluators = _resolve_evaluators(evaluators)
        context = copy_context()
        experiment_results = context.run(self._score, evaluators, max_concurrency=max_concurrency)
        r1, r2, r3 = itertools.tee(experiment_results, 3)
        return _ExperimentManager((result['example'] for result in r1), experiment=self._experiment, metadata=self._metadata, client=self.client, runs=(result['run'] for result in r2), evaluation_results=(result['evaluation_results'] for result in r3), summary_results=self._summary_results)

    def with_summary_evaluators(self, summary_evaluators: Sequence[SUMMARY_EVALUATOR_T]) -> _ExperimentManager:
        """Lazily apply the provided summary evaluators to the experiment."""
        wrapped_evaluators = _wrap_summary_evaluators(summary_evaluators)
        context = copy_context()
        aggregate_feedback_gen = context.run(self._apply_summary_evaluators, wrapped_evaluators)
        return _ExperimentManager(self.examples, experiment=self._experiment, metadata=self._metadata, client=self.client, runs=self.runs, evaluation_results=self._evaluation_results, summary_results=aggregate_feedback_gen)

    def get_results(self) -> Iterable[ExperimentResultRow]:
        """Return the traces, evaluation results, and associated examples."""
        for run, example, evaluation_results in zip(self.runs, self.examples, self.evaluation_results):
            yield ExperimentResultRow(run=run, example=example, evaluation_results=evaluation_results)

    def get_summary_scores(self) -> Dict[str, List[dict]]:
        """If summary_evaluators were applied, consume and return the results."""
        if self._summary_results is None:
            return {'results': []}
        return {'results': [res for results in self._summary_results for res in results['results']]}

    def _predict(self, target: TARGET_T, /, max_concurrency: Optional[int]=None) -> Generator[_ForwardResults, None, None]:
        """Run the target function on the examples."""
        fn = _ensure_traceable(target)
        if max_concurrency == 0:
            for example in self.examples:
                yield _forward(fn, example, self.experiment_name, self._metadata, self.client)
        else:
            with cf.ThreadPoolExecutor(max_concurrency) as executor:
                futures = [executor.submit(_forward, fn, example, self.experiment_name, self._metadata, self.client) for example in self.examples]
                for future in cf.as_completed(futures):
                    yield future.result()
        self._end()

    def _run_evaluators(self, evaluators: Sequence[RunEvaluator], current_results: ExperimentResultRow) -> ExperimentResultRow:
        current_context = rh.get_tracing_context()
        metadata = {**(current_context['metadata'] or {}), **{'experiment': self.experiment_name, 'reference_example_id': current_results['example'].id, 'reference_run_id': current_results['run'].id}}
        with rh.tracing_context(**{**current_context, 'project_name': 'evaluators', 'metadata': metadata}):
            run = current_results['run']
            example = current_results['example']
            eval_results = current_results['evaluation_results']
            for evaluator in evaluators:
                try:
                    evaluator_response = evaluator.evaluate_run(run=run, example=example)
                    eval_results['results'].extend(self.client._log_evaluation_feedback(evaluator_response, run=run))
                except Exception as e:
                    logger.error(f'Error running evaluator {repr(evaluator)} on run {run.id}: {repr(e)}', exc_info=True)
            return ExperimentResultRow(run=run, example=example, evaluation_results=eval_results)

    def _score(self, evaluators: Sequence[RunEvaluator], max_concurrency: Optional[int]=None) -> Iterable[ExperimentResultRow]:
        """Run the evaluators on the prediction stream.

        Expects runs to be available in the manager.
        (e.g. from a previous prediction step)
        """
        if max_concurrency == 0:
            for current_results in self.get_results():
                yield self._run_evaluators(evaluators, current_results)
        else:
            with cf.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = []
                for current_results in self.get_results():
                    futures.append(executor.submit(self._run_evaluators, evaluators, current_results))
                for future in cf.as_completed(futures):
                    result = future.result()
                    yield result

    def _apply_summary_evaluators(self, summary_evaluators: Sequence[SUMMARY_EVALUATOR_T]) -> Generator[EvaluationResults, None, None]:
        runs, examples = ([], [])
        for run, example in zip(self.runs, self.examples):
            runs.append(run)
            examples.append(example)
        aggregate_feedback = []
        with cf.ThreadPoolExecutor() as executor:
            project_id = self._get_experiment().id
            current_context = rh.get_tracing_context()
            metadata = {**(current_context['metadata'] or {}), **{'experiment': self.experiment_name, 'experiment_id': project_id}}
            with rh.tracing_context(**{**current_context, 'project_name': 'evaluators', 'metadata': metadata}):
                for evaluator in summary_evaluators:
                    try:
                        summary_eval_result = evaluator(runs, examples)
                        flattened_results = self.client._select_eval_results(summary_eval_result, fn_name=evaluator.__name__)
                        aggregate_feedback.extend(flattened_results)
                        for result in flattened_results:
                            feedback = result.dict(exclude={'target_run_id'})
                            evaluator_info = feedback.pop('evaluator_info', None)
                            executor.submit(self.client.create_feedback, **feedback, run_id=None, project_id=project_id, source_info=evaluator_info)
                    except Exception as e:
                        logger.error(f'Error running summary evaluator {repr(evaluator)}: {e}')
        yield {'results': aggregate_feedback}

    def _get_dataset_version(self) -> Optional[str]:
        examples = list(self.examples)
        modified_at = [ex.modified_at for ex in examples if ex.modified_at]
        max_modified_at = max(modified_at) if modified_at else None
        return max_modified_at.isoformat() if max_modified_at else None

    def _end(self) -> None:
        experiment = self._experiment
        if experiment is None:
            raise ValueError('Experiment not started yet.')
        project_metadata = self._get_experiment_metadata()
        project_metadata['dataset_version'] = self._get_dataset_version()
        self.client.update_project(experiment.id, end_time=datetime.datetime.now(datetime.timezone.utc), metadata=project_metadata)