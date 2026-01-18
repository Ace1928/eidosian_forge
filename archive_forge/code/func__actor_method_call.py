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
def _actor_method_call(self, method_name: str, args: List[Any]=None, kwargs: Dict[str, Any]=None, name: str='', num_returns: Optional[int]=None, max_retries: int=None, retry_exceptions: Union[bool, list, tuple]=None, concurrency_group_name: Optional[str]=None, generator_backpressure_num_objects: Optional[int]=None):
    """Method execution stub for an actor handle.

        This is the function that executes when
        `actor.method_name.remote(*args, **kwargs)` is called. Instead of
        executing locally, the method is packaged as a task and scheduled
        to the remote actor instance.

        Args:
            method_name: The name of the actor method to execute.
            args: A list of arguments for the actor method.
            kwargs: A dictionary of keyword arguments for the actor method.
            name: The name to give the actor method call task.
            num_returns: The number of return values for the method.
            max_retries: Number of retries when method fails.
            retry_exceptions: Boolean of whether you want to retry all user-raised
                exceptions, or a list of allowlist exceptions to retry.

        Returns:
            object_refs: A list of object refs returned by the remote actor
                method.
        """
    worker = ray._private.worker.global_worker
    args = args or []
    kwargs = kwargs or {}
    if self._ray_is_cross_language:
        list_args = cross_language._format_args(worker, args, kwargs)
        function_descriptor = cross_language._get_function_descriptor_for_actor_method(self._ray_actor_language, self._ray_actor_creation_function_descriptor, method_name, signature=str(len(args) + len(kwargs)))
    else:
        function_signature = self._ray_method_signatures[method_name]
        if not args and (not kwargs) and (not function_signature):
            list_args = []
        else:
            list_args = signature.flatten_args(function_signature, args, kwargs)
        function_descriptor = self._ray_function_descriptor[method_name]
    if worker.mode == ray.LOCAL_MODE:
        assert not self._ray_is_cross_language, 'Cross language remote actor method cannot be executed locally.'
    if num_returns == 'dynamic':
        num_returns = -1
    elif num_returns == 'streaming':
        num_returns = ray._raylet.STREAMING_GENERATOR_RETURN
    retry_exception_allowlist = None
    if retry_exceptions is None:
        retry_exceptions = False
    elif isinstance(retry_exceptions, (list, tuple)):
        retry_exception_allowlist = tuple(retry_exceptions)
        retry_exceptions = True
    assert isinstance(retry_exceptions, bool), 'retry_exceptions can either be             boolean or list/tuple of exception types.'
    if generator_backpressure_num_objects is None:
        generator_backpressure_num_objects = -1
    object_refs = worker.core_worker.submit_actor_task(self._ray_actor_language, self._ray_actor_id, function_descriptor, list_args, name, num_returns, max_retries, retry_exceptions, retry_exception_allowlist, self._ray_actor_method_cpus, concurrency_group_name if concurrency_group_name is not None else b'', generator_backpressure_num_objects)
    if num_returns == STREAMING_GENERATOR_RETURN:
        assert len(object_refs) == 1
        generator_ref = object_refs[0]
        return ObjectRefGenerator(generator_ref, worker)
    if len(object_refs) == 1:
        object_refs = object_refs[0]
    elif len(object_refs) == 0:
        object_refs = None
    return object_refs