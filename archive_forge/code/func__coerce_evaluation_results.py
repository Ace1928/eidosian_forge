import asyncio
import inspect
import uuid
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, cast
from typing_extensions import TypedDict
from functools import wraps
from langsmith.schemas import SCORE_TYPE, VALUE_TYPE, Example, Run
def _coerce_evaluation_results(self, results: Union[dict, EvaluationResults], source_run_id: uuid.UUID) -> Union[EvaluationResult, EvaluationResults]:
    if 'results' in results:
        cp = results.copy()
        cp['results'] = [self._coerce_evaluation_result(r, source_run_id=source_run_id) for r in results['results']]
        return EvaluationResults(**cp)
    return self._coerce_evaluation_result(cast(dict, results), allow_no_key=True, source_run_id=source_run_id)