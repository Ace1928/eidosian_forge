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
class _ActorClassMethodMetadata(object):
    """Metadata for all methods in an actor class. This data can be cached.

    Attributes:
        methods: The actor methods.
        decorators: Optional decorators that should be applied to the
            method invocation function before invoking the actor methods. These
            can be set by attaching the attribute
            "__ray_invocation_decorator__" to the actor method.
        signatures: The signatures of the methods.
        num_returns: The default number of return values for
            each actor method.
        max_retries: Number of retries on method failure.
        retry_exceptions: Boolean of whether you want to retry all user-raised
            exceptions, or a list of allowlist exceptions to retry, for each method.
    """
    _cache = {}

    def __init__(self):
        class_name = type(self).__name__
        raise TypeError(f"{class_name} can not be constructed directly, instead of running '{class_name}()', try '{class_name}.create()'")

    @classmethod
    def reset_cache(cls):
        cls._cache.clear()

    @classmethod
    def create(cls, modified_class, actor_creation_function_descriptor):
        cached_meta = cls._cache.get(actor_creation_function_descriptor)
        if cached_meta is not None:
            return cached_meta
        self = cls.__new__(cls)
        actor_methods = inspect.getmembers(modified_class, is_function_or_method)
        self.methods = dict(actor_methods)
        self.decorators = {}
        self.signatures = {}
        self.num_returns = {}
        self.max_retries = {}
        self.retry_exceptions = {}
        self.method_is_generator = {}
        self.generator_backpressure_num_objects = {}
        self.concurrency_group_for_methods = {}
        for method_name, method in actor_methods:
            method = inspect.unwrap(method)
            is_bound = is_class_method(method) or is_static_method(modified_class, method_name)
            self.signatures[method_name] = signature.extract_signature(method, ignore_first=not is_bound)
            if hasattr(method, '__ray_num_returns__'):
                self.num_returns[method_name] = method.__ray_num_returns__
            else:
                self.num_returns[method_name] = None
            if hasattr(method, '__ray_max_retries__'):
                self.max_retries[method_name] = method.__ray_max_retries__
            if hasattr(method, '__ray_retry_exceptions__'):
                self.retry_exceptions[method_name] = method.__ray_retry_exceptions__
            if hasattr(method, '__ray_invocation_decorator__'):
                self.decorators[method_name] = method.__ray_invocation_decorator__
            if hasattr(method, '__ray_concurrency_group__'):
                self.concurrency_group_for_methods[method_name] = method.__ray_concurrency_group__
            is_generator = inspect.isgeneratorfunction(method) or inspect.isasyncgenfunction(method)
            self.method_is_generator[method_name] = is_generator
            if hasattr(method, '__ray_generator_backpressure_num_objects__'):
                self.generator_backpressure_num_objects[method_name] = method.__ray_generator_backpressure_num_objects__
        cls._cache[actor_creation_function_descriptor] = self
        return self