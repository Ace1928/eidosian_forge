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
class WorkerTasks(object):
    context: Dict[str, Any] = {}
    functions: List[Callable] = []
    cronjobs: List[Dict] = []
    dependencies: Dict[str, Tuple[Union[Any, Callable], Dict]] = {}
    context_funcs: Dict[str, Callable] = {}
    startup_funcs: Dict[str, Tuple[Union[Any, Callable], Dict]] = {}
    shutdown_funcs: Dict[str, Tuple[Union[Any, Callable], Dict]] = {}
    silenced_functions: List[str] = []
    silenced_functions_by_stage: Dict[str, List[str]] = {'enqueue': [], 'dequeue': [], 'process': [], 'finish': [], 'sweep': [], 'retry': [], 'abort': []}
    queue_func: Union[Callable, 'TaskQueue'] = None