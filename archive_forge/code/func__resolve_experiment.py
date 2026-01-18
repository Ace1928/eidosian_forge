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
def _resolve_experiment(experiment: Optional[schemas.TracerSession], runs: Optional[Iterable[schemas.Run]], client: langsmith.Client) -> Tuple[Optional[Union[schemas.TracerSession, str]], Optional[Iterable[schemas.Run]]]:
    if experiment is not None:
        if not experiment.name:
            raise ValueError('Experiment name must be defined if provided.')
        return (experiment, None)
    if runs is not None:
        if runs is not None:
            runs_, runs = itertools.tee(runs)
            first_run = next(runs_)
        experiment = client.read_project(project_id=first_run.session_id)
        if not experiment.name:
            raise ValueError('Experiment name not found for provided runs.')
        return (experiment, runs)
    return (None, None)