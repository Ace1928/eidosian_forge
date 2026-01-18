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
def _print_experiment_start(self, project: schemas.TracerSession, first_example: schemas.Example) -> None:
    if project.url:
        project_url = project.url.split('?')[0]
        dataset_id = first_example.dataset_id
        base_url = project_url.split('/projects/p/')[0]
        comparison_url = f'{base_url}/datasets/{dataset_id}/compare?selectedSessions={project.id}'
        print(f"View the evaluation results for experiment: '{self.experiment_name}' at:\n{comparison_url}\n\n")
    else:
        print('Starting evaluation of experiment: %s', self.experiment_name)