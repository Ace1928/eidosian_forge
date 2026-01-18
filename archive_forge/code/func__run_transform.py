from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def _run_transform(self, run: Union[ls_schemas.Run, dict, ls_schemas.RunLikeDict], update: bool=False) -> dict:
    """Transform the given run object into a dictionary representation.

        Args:
            run (Union[ls_schemas.Run, dict]): The run object to transform.

        Returns:
            dict: The transformed run object as a dictionary.
        """
    if hasattr(run, 'dict') and callable(getattr(run, 'dict')):
        run_create: dict = run.dict()
    else:
        run_create = cast(dict, run)
    if 'id' not in run_create:
        run_create['id'] = uuid.uuid4()
    elif isinstance(run_create['id'], str):
        run_create['id'] = uuid.UUID(run_create['id'])
    if 'inputs' in run_create and run_create['inputs'] is not None:
        run_create['inputs'] = self._hide_run_inputs(run_create['inputs'])
    if 'outputs' in run_create and run_create['outputs'] is not None:
        run_create['outputs'] = self._hide_run_outputs(run_create['outputs'])
    if not update and (not run_create.get('start_time')):
        run_create['start_time'] = datetime.datetime.now(datetime.timezone.utc)
    return run_create