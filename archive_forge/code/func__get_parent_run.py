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
def _get_parent_run(langsmith_extra: LangSmithExtra) -> Optional[run_trees.RunTree]:
    parent = langsmith_extra.get('parent')
    if isinstance(parent, run_trees.RunTree):
        return parent
    if isinstance(parent, dict):
        return run_trees.RunTree.from_headers(parent)
    if isinstance(parent, str):
        return run_trees.RunTree.from_dotted_order(parent)
    run_tree = langsmith_extra.get('run_tree')
    if run_tree:
        return run_tree
    return get_current_run_tree()