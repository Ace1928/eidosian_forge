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
def _make_actor(cls, actor_options):
    Class = _modify_class(cls)
    _inject_tracing_into_class(Class)
    if 'max_restarts' in actor_options:
        if actor_options['max_restarts'] != -1:
            actor_options['max_restarts'] = min(actor_options['max_restarts'], ray_constants.MAX_INT64_VALUE)
    return ActorClass._ray_from_modified_class(Class, ActorClassID.from_random(), actor_options)