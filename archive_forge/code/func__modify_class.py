import inspect
import logging
import weakref
from typing import Any, Dict, List, Optional, Union
import ray._private.ray_constants as ray_constants
import ray._private.signature as signature
import ray._private.worker
import ray._raylet
from ray import ActorClassID, Language, cross_language
from ray._private import ray_option_utils
from ray._private.async_compat import is_async_func
from ray._private.auto_init_hook import wrap_auto_init
from ray._private.client_mode_hook import (
from ray._private.inspect_util import (
from ray._private.ray_option_utils import _warn_if_using_deprecated_placement_group
from ray._private.utils import get_runtime_env_info, parse_runtime_env
from ray._raylet import (
from ray.exceptions import AsyncioActorExit
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.placement_group import _configure_placement_group_based_on_context
from ray.util.scheduling_strategies import (
from ray.util.tracing.tracing_helper import (
def _modify_class(cls):
    if hasattr(cls, '__ray_actor_class__'):
        return cls
    if not issubclass(cls, object):
        raise TypeError("The @ray.remote decorator cannot be applied to old-style classes. In Python 2, you must declare the class with 'class ClassName(object):' instead of 'class ClassName:'.")

    class Class(cls):
        __ray_actor_class__ = cls

        def __ray_ready__(self):
            return True

        def __ray_call__(self, fn, *args, **kwargs):
            return fn(self, *args, **kwargs)

        def __ray_terminate__(self):
            worker = ray._private.worker.global_worker
            if worker.mode != ray.LOCAL_MODE:
                ray.actor.exit_actor()
    Class.__module__ = cls.__module__
    Class.__name__ = cls.__name__
    if not is_function_or_method(getattr(Class, '__init__', None)):

        def __init__(self):
            pass
        Class.__init__ = __init__
    return Class