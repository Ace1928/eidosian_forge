import os
import re
import json
import socket
import contextlib
import functools
from lazyops.utils.helpers import is_coro_func
from lazyops.utils.logs import default_logger as logger
from typing import Optional, Dict, Any, Union, Callable, List, Tuple, TYPE_CHECKING
from aiokeydb.v2.types import BaseSettings, validator, lazyproperty, KeyDBUri
from aiokeydb.v2.types.static import TaskType
from aiokeydb.v2.serializers import SerializerType
from aiokeydb.v2.utils.queue import run_in_executor
from aiokeydb.v2.utils.cron import validate_cron_schedule
def add_function_to_silenced(self, name: str, silenced_stages: Optional[List[str]]=None, **kwargs):
    """
        Adds a function to the silenced functions
        """
    if silenced_stages:
        for stage in silenced_stages:
            if stage not in self.tasks.silenced_functions_by_stage:
                continue
            if name not in self.tasks.silenced_functions_by_stage[stage]:
                self.tasks.silenced_functions_by_stage[stage].append(name)
    elif name not in self.tasks.silenced_functions:
        self.tasks.silenced_functions.append(name)