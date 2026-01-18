import inspect
import os
import sys
import threading
import warnings
from collections import UserDict, defaultdict, deque
from datetime import datetime
from datetime import timezone as datetime_timezone
from operator import attrgetter
from click.exceptions import Exit
from dateutil.parser import isoparse
from kombu import pools
from kombu.clocks import LamportClock
from kombu.common import oid_from
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from kombu.utils.uuid import uuid
from vine import starpromise
from celery import platforms, signals
from celery._state import (_announce_app_finalized, _deregister_app, _register_app, _set_current_app, _task_stack,
from celery.exceptions import AlwaysEagerIgnored, ImproperlyConfigured
from celery.loaders import get_loader_cls
from celery.local import PromiseProxy, maybe_evaluate
from celery.utils import abstract
from celery.utils.collections import AttributeDictMixin
from celery.utils.dispatch import Signal
from celery.utils.functional import first, head_from_fun, maybe_list
from celery.utils.imports import gen_task_name, instantiate, symbol_by_name
from celery.utils.log import get_logger
from celery.utils.objects import FallbackContext, mro_lookup
from celery.utils.time import maybe_make_aware, timezone, to_utc
from . import backends, builtins  # noqa
from .annotations import prepare as prepare_annotations
from .autoretry import add_autoretry_behaviour
from .defaults import DEFAULT_SECURITY_DIGEST, find_deprecated_settings
from .registry import TaskRegistry
from .utils import (AppPickler, Settings, _new_key_to_old, _old_key_to_new, _unpickle_app, _unpickle_app_v2, appstr,
def _task_from_fun(self, fun, name=None, base=None, bind=False, **options):
    if not self.finalized and (not self.autofinalize):
        raise RuntimeError('Contract breach: app not finalized')
    name = name or self.gen_task_name(fun.__name__, fun.__module__)
    base = base or self.Task
    if name not in self._tasks:
        run = fun if bind else staticmethod(fun)
        task = type(fun.__name__, (base,), dict({'app': self, 'name': name, 'run': run, '_decorated': True, '__doc__': fun.__doc__, '__module__': fun.__module__, '__annotations__': fun.__annotations__, '__header__': self.type_checker(fun, bound=bind), '__wrapped__': run}, **options))()
        try:
            task.__qualname__ = fun.__qualname__
        except AttributeError:
            pass
        self._tasks[task.name] = task
        task.bind(self)
        add_autoretry_behaviour(task, **options)
    else:
        task = self._tasks[name]
    return task