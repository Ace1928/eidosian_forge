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
def _get_dataset_version(self) -> Optional[str]:
    examples = list(self.examples)
    modified_at = [ex.modified_at for ex in examples if ex.modified_at]
    max_modified_at = max(modified_at) if modified_at else None
    return max_modified_at.isoformat() if max_modified_at else None