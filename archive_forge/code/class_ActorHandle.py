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
class ActorHandle:
    """A handle to an actor.

    The fields in this class are prefixed with _ray_ to hide them from the user
    and to avoid collision with actor method names.

    An ActorHandle can be created in three ways. First, by calling .remote() on
    an ActorClass. Second, by passing an actor handle into a task (forking the
    ActorHandle). Third, by directly serializing the ActorHandle (e.g., with
    cloudpickle).

    Attributes:
        _ray_actor_language: The actor language.
        _ray_actor_id: Actor ID.
        _ray_method_is_generator: Map of method name -> if it is a generator
            method.
        _ray_method_decorators: Optional decorators for the function
            invocation. This can be used to change the behavior on the
            invocation side, whereas a regular decorator can be used to change
            the behavior on the execution side.
        _ray_method_signatures: The signatures of the actor methods.
        _ray_method_max_retries: Max number of retries on method failure.
        _ray_method_num_returns: The default number of return values for
            each method.
        _ray_method_retry_exceptions: The default value of boolean of whether you want
            to retry all user-raised exceptions, or a list of allowlist exceptions to
            retry.
        _ray_method_generator_backpressure_num_objects: Generator-only
            config. The max number of objects to generate before it
            starts pausing a generator.
        _ray_actor_method_cpus: The number of CPUs required by actor methods.
        _ray_original_handle: True if this is the original actor handle for a
            given actor. If this is true, then the actor will be destroyed when
            this handle goes out of scope.
        _ray_is_cross_language: Whether this actor is cross language.
        _ray_actor_creation_function_descriptor: The function descriptor
            of the actor creation task.
    """

    def __init__(self, language, actor_id, max_task_retries: Optional[int], method_is_generator: Dict[str, bool], method_decorators, method_signatures, method_num_returns: Dict[str, int], method_max_retries: Dict[str, int], method_retry_exceptions: Dict[str, Union[bool, list, tuple]], method_generator_backpressure_num_objects: Dict[str, int], actor_method_cpus: int, actor_creation_function_descriptor, session_and_job, original_handle=False):
        self._ray_actor_language = language
        self._ray_actor_id = actor_id
        self._ray_max_task_retries = max_task_retries
        self._ray_original_handle = original_handle
        self._ray_method_is_generator = method_is_generator
        self._ray_method_decorators = method_decorators
        self._ray_method_signatures = method_signatures
        self._ray_method_num_returns = method_num_returns
        self._ray_method_max_retries = method_max_retries
        self._ray_method_retry_exceptions = method_retry_exceptions
        self._ray_method_generator_backpressure_num_objects = method_generator_backpressure_num_objects
        self._ray_actor_method_cpus = actor_method_cpus
        self._ray_session_and_job = session_and_job
        self._ray_is_cross_language = language != Language.PYTHON
        self._ray_actor_creation_function_descriptor = actor_creation_function_descriptor
        self._ray_function_descriptor = {}
        if not self._ray_is_cross_language:
            assert isinstance(actor_creation_function_descriptor, PythonFunctionDescriptor)
            module_name = actor_creation_function_descriptor.module_name
            class_name = actor_creation_function_descriptor.class_name
            for method_name in self._ray_method_signatures.keys():
                function_descriptor = PythonFunctionDescriptor(module_name, method_name, class_name)
                self._ray_function_descriptor[method_name] = function_descriptor
                method = ActorMethod(self, method_name, self._ray_method_num_returns[method_name], self._ray_method_max_retries.get(method_name, self._ray_max_task_retries) or 0, self._ray_method_retry_exceptions.get(method_name), self._ray_method_is_generator[method_name], self._ray_method_generator_backpressure_num_objects.get(method_name), decorator=self._ray_method_decorators.get(method_name))
                setattr(self, method_name, method)

    def __del__(self):
        try:
            if ray._private.worker:
                worker = ray._private.worker.global_worker
                if worker.connected and hasattr(worker, 'core_worker'):
                    worker.core_worker.remove_actor_handle_reference(self._ray_actor_id)
        except AttributeError:
            pass

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

    def __getattr__(self, item):
        if not self._ray_is_cross_language:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
        if item in ['__ray_terminate__']:

            class FakeActorMethod(object):

                def __call__(self, *args, **kwargs):
                    raise TypeError("Actor methods cannot be called directly. Instead of running 'object.{}()', try 'object.{}.remote()'.".format(item, item))

                def remote(self, *args, **kwargs):
                    logger.warning(f'Actor method {item} is not supported by cross language.')
            return FakeActorMethod()
        return ActorMethod(self, item, ray_constants.DEFAULT_ACTOR_METHOD_NUM_RETURN_VALS, 0, False, False, self._ray_method_generator_backpressure_num_objects.get(item, -1), decorator=None)

    def __dir__(self):
        return self._ray_method_signatures.keys()

    def __repr__(self):
        return f'Actor({self._ray_actor_creation_function_descriptor.class_name}, {self._actor_id.hex()})'

    @property
    def _actor_id(self):
        return self._ray_actor_id

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

    @classmethod
    def _deserialization_helper(cls, state, outer_object_ref=None):
        """This is defined in order to make pickling work.

        Args:
            state: The serialized state of the actor handle.
            outer_object_ref: The ObjectRef that the serialized actor handle
                was contained in, if any. This is used for counting references
                to the actor handle.

        """
        worker = ray._private.worker.global_worker
        worker.check_connected()
        if hasattr(worker, 'core_worker'):
            return worker.core_worker.deserialize_and_register_actor_handle(state, outer_object_ref)
        else:
            return cls(state['actor_language'], state['actor_id'], state['max_task_retries'], state['method_is_generator'], state['method_decorators'], state['method_signatures'], state['method_num_returns'], state['method_max_retries'], state['method_retry_exceptions'], state['method_generator_backpressure_num_objects'], state['actor_method_cpus'], state['actor_creation_function_descriptor'], worker.current_session_and_job)

    def __reduce__(self):
        """This code path is used by pickling but not by Ray forking."""
        serialized, _ = self._serialization_helper()
        return (ActorHandle._deserialization_helper, (serialized, None))