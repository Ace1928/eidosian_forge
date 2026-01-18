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
def _serialization_helper(self):
    """This is defined in order to make pickling work.

        Returns:
            A dictionary of the information needed to reconstruct the object.
        """
    worker = ray._private.worker.global_worker
    worker.check_connected()
    if hasattr(worker, 'core_worker'):
        state = worker.core_worker.serialize_actor_handle(self._ray_actor_id)
    else:
        state = ({'actor_language': self._ray_actor_language, 'actor_id': self._ray_actor_id, 'max_task_retries': self._ray_max_task_retries, 'method_is_generator': self._ray_method_is_generator, 'method_decorators': self._ray_method_decorators, 'method_signatures': self._ray_method_signatures, 'method_num_returns': self._ray_method_num_returns, 'method_max_retries': self._ray_method_max_retries, 'method_retry_exceptions': self._ray_method_retry_exceptions, 'method_generator_backpressure_num_objects': self._ray_method_generator_backpressure_num_objects, 'actor_method_cpus': self._ray_actor_method_cpus, 'actor_creation_function_descriptor': self._ray_actor_creation_function_descriptor}, None)
    return state