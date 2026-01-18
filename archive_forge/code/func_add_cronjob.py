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
def add_cronjob(self, schedule: Optional[Union[Dict, List, str]]=None, _fx: Optional[Callable]=None, name: Optional[str]=None, verbose: Optional[bool]=None, silenced: Optional[bool]=None, silenced_stages: Optional[List[str]]=None, default_kwargs: Optional[dict]=None, callback: Optional[Union[str, Callable]]=None, callback_kwargs: Optional[dict]=None, disabled: Optional[bool]=False, **kwargs):
    """
        Adds a function to `WorkerTask.cronjobs`.
        WorkerCronFuncs = {
            {'coroutine': refresh_spot_data, 'name': 'refresh_spot_data', 'minute': {10, 30, 50}},
        }
        """
    if verbose is None:
        verbose = self.debug_enabled
    if schedule and isinstance(schedule, str):
        schedule = validate_cron_schedule(schedule)
    if _fx is not None:
        if disabled is True:
            return
        cron = {'function': _fx, 'cron_name': name, 'default_kwargs': default_kwargs, 'cron': schedule, 'silenced': silenced, 'callback': callback, **kwargs}
        if callback_kwargs:
            cron['callback_kwargs'] = callback_kwargs
        self.tasks.cronjobs.append(cron)
        if silenced or silenced_stages:
            self.add_function_to_silenced(name or _fx.__qualname__, silenced_stages=silenced_stages)
        if verbose:
            logger.info(f'Registered CronJob: {cron}')
        return

    def decorator(func: Callable):
        nonlocal schedule
        if disabled is True:
            return func
        cron = {'function': func, 'cron': schedule, 'cron_name': name, 'default_kwargs': default_kwargs, 'silenced': silenced, 'callback': callback, **kwargs}
        if callback_kwargs:
            cron['callback_kwargs'] = callback_kwargs
        self.tasks.cronjobs.append(cron)
        if silenced or silenced_stages:
            self.add_function_to_silenced(name or func.__qualname__, silenced_stages=silenced_stages)
        if verbose:
            logger.info(f'Registered CronJob: {cron}')
        return func
    return decorator