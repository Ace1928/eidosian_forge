from __future__ import annotations
import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import traceback
import uuid
import warnings
from contextvars import copy_context
from typing import (
from langsmith import client as ls_client
from langsmith import run_trees, utils
from langsmith._internal import _aiter as aitertools
class LangSmithExtra(TypedDict, total=False):
    """Any additional info to be injected into the run dynamically."""
    reference_example_id: Optional[ls_client.ID_TYPE]
    run_extra: Optional[Dict]
    parent: Optional[Union[run_trees.RunTree, str, Mapping]]
    run_tree: Optional[run_trees.RunTree]
    project_name: Optional[str]
    metadata: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    run_id: Optional[ls_client.ID_TYPE]
    client: Optional[ls_client.Client]
    on_end: Optional[Callable[[run_trees.RunTree], Any]]