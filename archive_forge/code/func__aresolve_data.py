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
def _aresolve_data(data: Union[DATA_T, AsyncIterable[schemas.Example]], *, client: langsmith.Client) -> AsyncIterator[schemas.Example]:
    """Return the examples for the given dataset."""
    if isinstance(data, AsyncIterable):
        return aitertools.ensure_async_iterator(data)
    return aitertools.ensure_async_iterator(_resolve_data(data, client=client))