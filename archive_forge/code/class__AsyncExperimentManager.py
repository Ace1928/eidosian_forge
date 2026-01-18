from __future__ import annotations
import asyncio
import datetime
import logging
import pathlib
import uuid
from typing import (
import langsmith
from langsmith import run_helpers as rh
from langsmith import run_trees, schemas
from langsmith import utils as ls_utils
from langsmith._internal import _aiter as aitertools
from langsmith.beta import warn_beta
from langsmith.evaluation._runner import (
from langsmith.evaluation.evaluator import EvaluationResults, RunEvaluator
class _AsyncExperimentManager(_ExperimentManagerMixin):
    """Manage the execution of experiments asynchronously.

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

    def __init__(self, data: Union[DATA_T, AsyncIterable[schemas.Example]], /, experiment: Optional[Union[schemas.TracerSession, str]]=None, metadata: Optional[dict]=None, runs: Optional[Union[Iterable[schemas.Run], AsyncIterable[schemas.Run]]]=None, client: Optional[langsmith.Client]=None, evaluation_results: Optional[AsyncIterable[EvaluationResults]]=None, summary_results: Optional[AsyncIterable[EvaluationResults]]=None):
        super().__init__(experiment=experiment, metadata=metadata, client=client)
        self._data = data
        self._examples: Optional[AsyncIterable[schemas.Example]] = None
        self._runs = aitertools.ensure_async_iterator(runs) if runs is not None else None
        self._evaluation_results = evaluation_results
        self._summary_results = summary_results

    def aget_examples(self) -> AsyncIterator[schemas.Example]:
        if self._examples is None:
            self._examples = _aresolve_data(self._data, client=self.client)
        self._examples, examples_iter = aitertools.atee(aitertools.ensure_async_iterator(self._examples), 2, lock=asyncio.Lock())
        return examples_iter

    async def get_dataset_id(self) -> str:
        if self._experiment is None or not getattr(self._experiment, 'reference_dataset_id', None):
            example = await aitertools.py_anext(self.aget_examples())
            if example is None:
                raise ValueError('No examples found in the dataset.')
            return str(example.dataset_id)
        return str(self._experiment.reference_dataset_id)

    async def aget_runs(self) -> AsyncIterator[schemas.Run]:
        if self._runs is None:
            raise ValueError('Runs not loaded yet.')
        self._runs, runs = aitertools.atee(aitertools.ensure_async_iterator(self._runs), 2, lock=asyncio.Lock())
        async for run in runs:
            yield run

    async def aget_evaluation_results(self) -> AsyncIterator[EvaluationResults]:
        if self._evaluation_results is None:
            async for _ in self.aget_examples():
                yield {'results': []}
        else:
            self._evaluation_results, evaluation_results = aitertools.atee(aitertools.ensure_async_iterator(self._evaluation_results), 2, lock=asyncio.Lock())
            async for result in evaluation_results:
                yield result

    async def astart(self) -> _AsyncExperimentManager:
        first_example = await aitertools.py_anext(self.aget_examples())
        if not first_example:
            raise ValueError('No examples found in the dataset.')
        project = self._get_project(first_example)
        self._print_experiment_start(project, first_example)
        return self.__class__(self.aget_examples(), experiment=project, metadata=self._metadata, client=self.client, runs=self._runs, evaluation_results=self._evaluation_results)

    async def awith_predictions(self, target: ATARGET_T, /, max_concurrency: Optional[int]=None) -> _AsyncExperimentManager:
        _experiment_results = self._apredict(target, max_concurrency=max_concurrency)
        r1, r2 = aitertools.atee(_experiment_results, 2, lock=asyncio.Lock())
        return _AsyncExperimentManager((pred['example'] async for pred in r1), experiment=self._experiment, metadata=self._metadata, client=self.client, runs=(pred['run'] async for pred in r2))

    async def awith_evaluators(self, evaluators: Sequence[Union[EVALUATOR_T, AEVALUATOR_T]], *, max_concurrency: Optional[int]=None) -> _AsyncExperimentManager:
        evaluators = _resolve_evaluators(evaluators)
        experiment_results = self._ascore(evaluators, max_concurrency=max_concurrency)
        r1, r2, r3 = aitertools.atee(experiment_results, 3, lock=asyncio.Lock())
        return _AsyncExperimentManager((result['example'] async for result in r1), experiment=self._experiment, metadata=self._metadata, client=self.client, runs=(result['run'] async for result in r2), evaluation_results=(result['evaluation_results'] async for result in r3), summary_results=self._summary_results)

    async def awith_summary_evaluators(self, summary_evaluators: Sequence[SUMMARY_EVALUATOR_T]) -> _AsyncExperimentManager:
        wrapped_evaluators = _wrap_summary_evaluators(summary_evaluators)
        aggregate_feedback_gen = self._aapply_summary_evaluators(wrapped_evaluators)
        return _AsyncExperimentManager(self.aget_examples(), experiment=self._experiment, metadata=self._metadata, client=self.client, runs=self.aget_runs(), evaluation_results=self._evaluation_results, summary_results=aggregate_feedback_gen)

    async def aget_results(self) -> AsyncIterator[ExperimentResultRow]:
        async for run, example, evaluation_results in aitertools.async_zip(self.aget_runs(), self.aget_examples(), self.aget_evaluation_results()):
            yield ExperimentResultRow(run=run, example=example, evaluation_results=evaluation_results)

    async def aget_summary_scores(self) -> Dict[str, List[dict]]:
        if self._summary_results is None:
            return {'results': []}
        return {'results': [res async for results in self._summary_results for res in results['results']]}

    async def _apredict(self, target: ATARGET_T, /, max_concurrency: Optional[int]=None) -> AsyncIterator[_ForwardResults]:
        fn = _ensure_async_traceable(target)

        async def predict_all():
            async for example in self.aget_examples():
                yield _aforward(fn, example, self.experiment_name, self._metadata, self.client)
        async for result in aitertools.aiter_with_concurrency(max_concurrency, predict_all()):
            yield result
        await self._aend()

    async def _ascore(self, evaluators: Sequence[RunEvaluator], max_concurrency: Optional[int]=None) -> AsyncIterator[ExperimentResultRow]:

        async def score_all():
            async for current_results in self.aget_results():
                yield self._arun_evaluators(evaluators, current_results)
        async for result in aitertools.aiter_with_concurrency(max_concurrency, score_all()):
            yield result

    async def _arun_evaluators(self, evaluators: Sequence[RunEvaluator], current_results: ExperimentResultRow) -> ExperimentResultRow:
        current_context = rh.get_tracing_context()
        metadata = {**(current_context['metadata'] or {}), **{'experiment': self.experiment_name}}
        with rh.tracing_context(**{**current_context, 'project_name': 'evaluators', 'metadata': metadata}):
            run = current_results['run']
            example = current_results['example']
            eval_results = current_results['evaluation_results']
            for evaluator in evaluators:
                try:
                    evaluator_response = await evaluator.aevaluate_run(run=run, example=example)
                    eval_results['results'].extend(self.client._log_evaluation_feedback(evaluator_response, run=run))
                except Exception as e:
                    logger.error(f'Error running evaluator {repr(evaluator)} on run {run.id}: {repr(e)}', exc_info=True)
            return ExperimentResultRow(run=run, example=example, evaluation_results=eval_results)

    async def _aapply_summary_evaluators(self, summary_evaluators: Sequence[SUMMARY_EVALUATOR_T]) -> AsyncIterator[EvaluationResults]:
        runs, examples = ([], [])
        async_examples = aitertools.ensure_async_iterator(self.aget_examples())
        async for run, example in aitertools.async_zip(self.aget_runs(), async_examples):
            runs.append(run)
            examples.append(example)
        aggregate_feedback = []
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
                        self.client.create_feedback(**feedback, run_id=None, project_id=project_id, source_info=evaluator_info)
                except Exception as e:
                    logger.error(f'Error running summary evaluator {repr(evaluator)}: {e}')
        yield {'results': aggregate_feedback}

    async def _get_dataset_version(self) -> Optional[str]:
        modified_at = []
        async for example in self.aget_examples():
            if example.modified_at:
                modified_at.append(example.modified_at)
        max_modified_at = max(modified_at) if modified_at else None
        return max_modified_at.isoformat() if max_modified_at else None

    async def _aend(self) -> None:
        experiment = self._experiment
        if experiment is None:
            raise ValueError('Experiment not started yet.')
        project_metadata = self._get_experiment_metadata()
        project_metadata['dataset_version'] = await self._get_dataset_version()
        self.client.update_project(experiment.id, end_time=datetime.datetime.now(datetime.timezone.utc), metadata=project_metadata)