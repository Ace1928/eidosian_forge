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
@PublicAPI
class ActorMethod:
    """A class used to invoke an actor method.

    Note: This class only keeps a weak ref to the actor, unless it has been
    passed to a remote function. This avoids delays in GC of the actor.

    Attributes:
        _actor_ref: A weakref handle to the actor.
        _method_name: The name of the actor method.
        _num_returns: The default number of return values that the method
            invocation should return. If None is given, it uses
            DEFAULT_ACTOR_METHOD_NUM_RETURN_VALS for a normal actor task
            and "streaming" for a generator task (when `is_generator` is True).
        _max_retries: [Internal] Number of retries on method failure.
        _retry_exceptions: Boolean of whether you want to retry all user-raised
            exceptions, or a list of allowlist exceptions to retry.
        _is_generator: True if a given method is a Python generator.
        _generator_backpressure_num_objects: Generator-only config.
            If a number of unconsumed objects reach this threshold,
            a actor task stop pausing.
        _decorator: An optional decorator that should be applied to the actor
            method invocation (as opposed to the actor method execution) before
            invoking the method. The decorator must return a function that
            takes in two arguments ("args" and "kwargs"). In most cases, it
            should call the function that was passed into the decorator and
            return the resulting ObjectRefs. For an example, see
            "test_decorated_method" in "python/ray/tests/test_actor.py".
    """

    def __init__(self, actor, method_name, num_returns: Optional[Union[int, str]], _max_retries: int, retry_exceptions: Union[bool, list, tuple], is_generator: bool, generator_backpressure_num_objects: int, decorator=None, hardref=False):
        self._actor_ref = weakref.ref(actor)
        self._method_name = method_name
        self._num_returns = num_returns
        if self._num_returns is None:
            if is_generator:
                self._num_returns = 'streaming'
            else:
                self._num_returns = ray_constants.DEFAULT_ACTOR_METHOD_NUM_RETURN_VALS
        self._max_retries = _max_retries
        self._retry_exceptions = retry_exceptions
        self._is_generator = is_generator
        self._generator_backpressure_num_objects = generator_backpressure_num_objects
        self._decorator = decorator
        if hardref:
            self._actor_hard_ref = actor
        else:
            self._actor_hard_ref = None

    def __call__(self, *args, **kwargs):
        raise TypeError(f"Actor methods cannot be called directly. Instead of running 'object.{self._method_name}()', try 'object.{self._method_name}.remote()'.")

    def remote(self, *args, **kwargs):
        return self._remote(args, kwargs)

    def options(self, **options):
        """Convenience method for executing an actor method call with options.

        Same arguments as func._remote(), but returns a wrapped function
        that a non-underscore .remote() can be called on.

        Examples:
            # The following two calls are equivalent.
            >>> actor.my_method._remote(args=[x, y], name="foo", num_returns=2)
            >>> actor.my_method.options(name="foo", num_returns=2).remote(x, y)
        """
        func_cls = self

        class FuncWrapper:

            def remote(self, *args, **kwargs):
                return func_cls._remote(args=args, kwargs=kwargs, **options)
        return FuncWrapper()

    @wrap_auto_init
    @_tracing_actor_method_invocation
    def _remote(self, args=None, kwargs=None, name='', num_returns=None, _max_retries=None, retry_exceptions=None, concurrency_group=None, _generator_backpressure_num_objects=None):
        if num_returns is None:
            num_returns = self._num_returns
        max_retries = _max_retries
        if max_retries is None:
            max_retries = self._max_retries
        if max_retries is None:
            max_retries = 0
        if retry_exceptions is None:
            retry_exceptions = self._retry_exceptions
        if _generator_backpressure_num_objects is None:
            _generator_backpressure_num_objects = self._generator_backpressure_num_objects

        def invocation(args, kwargs):
            actor = self._actor_hard_ref or self._actor_ref()
            if actor is None:
                raise RuntimeError('Lost reference to actor')
            return actor._actor_method_call(self._method_name, args=args, kwargs=kwargs, name=name, num_returns=num_returns, max_retries=max_retries, retry_exceptions=retry_exceptions, concurrency_group_name=concurrency_group, generator_backpressure_num_objects=_generator_backpressure_num_objects)
        if self._decorator is not None:
            invocation = self._decorator(invocation)
        return invocation(args, kwargs)

    def __getstate__(self):
        return {'actor': self._actor_ref(), 'method_name': self._method_name, 'num_returns': self._num_returns, 'max_retries': self._max_retries, 'retry_exceptions': self._retry_exceptions, 'decorator': self._decorator, 'is_generator': self._is_generator, 'generator_backpressure_num_objects': self._generator_backpressure_num_objects}

    def __setstate__(self, state):
        self.__init__(state['actor'], state['method_name'], state['num_returns'], state['max_retries'], state['retry_exceptions'], state['is_generator'], state['generator_backpressure_num_objects'], state['decorator'], hardref=True)