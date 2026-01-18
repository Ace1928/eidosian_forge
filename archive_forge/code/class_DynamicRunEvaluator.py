import asyncio
import inspect
import uuid
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, cast
from typing_extensions import TypedDict
from functools import wraps
from langsmith.schemas import SCORE_TYPE, VALUE_TYPE, Example, Run
class DynamicRunEvaluator(RunEvaluator):
    """A dynamic evaluator that wraps a function and transforms it into a `RunEvaluator`.

    This class is designed to be used with the `@run_evaluator` decorator, allowing
    functions that take a `Run` and an optional `Example` as arguments, and return
    an `EvaluationResult` or `EvaluationResults`, to be used as instances of `RunEvaluator`.

    Attributes:
        func (Callable): The function that is wrapped by this evaluator.
    """

    def __init__(self, func: Callable[[Run, Optional[Example]], Union[_RUNNABLE_OUTPUT, Awaitable[_RUNNABLE_OUTPUT]]], afunc: Optional[Callable[[Run, Optional[Example]], Awaitable[_RUNNABLE_OUTPUT]]]=None):
        """Initialize the DynamicRunEvaluator with a given function.

        Args:
            func (Callable): A function that takes a `Run` and an optional `Example` as
            arguments, and returns an `EvaluationResult` or `EvaluationResults`.
        """
        wraps(func)(self)
        from langsmith import run_helpers
        if afunc is not None:
            self.afunc = run_helpers.ensure_traceable(afunc)
            self._name = getattr(afunc, '__name__', 'DynamicRunEvaluator')
        if inspect.iscoroutinefunction(func):
            if afunc is not None:
                raise TypeError('Func was provided as a coroutine function, but afunc was also provided. If providing both, func should be a regular function to avoid ambiguity.')
            self.afunc = run_helpers.ensure_traceable(func)
            self._name = getattr(func, '__name__', 'DynamicRunEvaluator')
        else:
            self.func = cast(run_helpers.SupportsLangsmithExtra[_RUNNABLE_OUTPUT], run_helpers.ensure_traceable(func))
            self._name = getattr(func, '__name__', 'DynamicRunEvaluator')

    def _coerce_evaluation_result(self, result: Union[EvaluationResult, dict], source_run_id: uuid.UUID, allow_no_key: bool=False) -> EvaluationResult:
        if isinstance(result, EvaluationResult):
            if not result.source_run_id:
                result.source_run_id = source_run_id
            return result
        try:
            if 'key' not in result:
                if allow_no_key:
                    result['key'] = self._name
            return EvaluationResult(**{'source_run_id': source_run_id, **result})
        except ValidationError as e:
            raise ValueError(f"Expected an EvaluationResult object, or dict with a metric 'key' and optional 'score'; got {result}") from e

    def _coerce_evaluation_results(self, results: Union[dict, EvaluationResults], source_run_id: uuid.UUID) -> Union[EvaluationResult, EvaluationResults]:
        if 'results' in results:
            cp = results.copy()
            cp['results'] = [self._coerce_evaluation_result(r, source_run_id=source_run_id) for r in results['results']]
            return EvaluationResults(**cp)
        return self._coerce_evaluation_result(cast(dict, results), allow_no_key=True, source_run_id=source_run_id)

    def _format_result(self, result: Union[EvaluationResult, EvaluationResults, dict], source_run_id: uuid.UUID) -> Union[EvaluationResult, EvaluationResults]:
        if isinstance(result, EvaluationResult):
            if not result.source_run_id:
                result.source_run_id = source_run_id
            return result
        if not isinstance(result, dict):
            raise ValueError(f'Expected a dict, EvaluationResult, or EvaluationResults, got {result}')
        return self._coerce_evaluation_results(result, source_run_id)

    @property
    def is_async(self) -> bool:
        """Check if the evaluator function is asynchronous.

        Returns:
            bool: True if the evaluator function is asynchronous, False otherwise.
        """
        return hasattr(self, 'afunc')

    def evaluate_run(self, run: Run, example: Optional[Example]=None) -> Union[EvaluationResult, EvaluationResults]:
        """Evaluate a run using the wrapped function.

        This method directly invokes the wrapped function with the provided arguments.

        Args:
            run (Run): The run to be evaluated.
            example (Optional[Example]): An optional example to be used in the evaluation.

        Returns:
            Union[EvaluationResult, EvaluationResults]: The result of the evaluation.
        """
        if not hasattr(self, 'func'):
            running_loop = asyncio.get_event_loop()
            if running_loop.is_running():
                raise RuntimeError('Cannot call `evaluate_run` on an async run evaluator from within an running event loop. Use `aevaluate_run` instead.')
            else:
                return running_loop.run_until_complete(self.aevaluate_run(run, example))
        source_run_id = uuid.uuid4()
        metadata: Dict[str, Any] = {'target_run_id': run.id}
        if getattr(run, 'session_id', None):
            metadata['experiment'] = str(run.session_id)
        result = self.func(run, example, langsmith_extra={'run_id': source_run_id, 'metadata': metadata})
        return self._format_result(result, source_run_id)

    async def aevaluate_run(self, run: Run, example: Optional[Example]=None):
        """Evaluate a run asynchronously using the wrapped async function.

        This method directly invokes the wrapped async function with the
            provided arguments.

        Args:
            run (Run): The run to be evaluated.
            example (Optional[Example]): An optional example to be used
                in the evaluation.

        Returns:
            Union[EvaluationResult, EvaluationResults]: The result of the evaluation.
        """
        if not hasattr(self, 'afunc'):
            return await super().aevaluate_run(run, example)
        source_run_id = uuid.uuid4()
        metadata: Dict[str, Any] = {'target_run_id': run.id}
        if getattr(run, 'session_id', None):
            metadata['experiment'] = str(run.session_id)
        result = await self.afunc(run, example, langsmith_extra={'run_id': source_run_id, 'metadata': metadata})
        return self._format_result(result, source_run_id)

    def __call__(self, run: Run, example: Optional[Example]=None) -> Union[EvaluationResult, EvaluationResults]:
        """Make the evaluator callable, allowing it to be used like a function.

        This method enables the evaluator instance to be called directly, forwarding the
        call to `evaluate_run`.

        Args:
            run (Run): The run to be evaluated.
            example (Optional[Example]): An optional example to be used in the evaluation.

        Returns:
            Union[EvaluationResult, EvaluationResults]: The result of the evaluation.
        """
        return self.evaluate_run(run, example)

    def __repr__(self) -> str:
        """Represent the DynamicRunEvaluator object."""
        return f'<DynamicRunEvaluator {getattr(self.func, '__name__')}>'