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
class ActorClass:
    """An actor class.

    This is a decorated class. It can be used to create actors.

    Attributes:
        __ray_metadata__: Contains metadata for the actor.
    """

    def __init__(cls, name, bases, attr):
        """Prevents users from directly inheriting from an ActorClass.

        This will be called when a class is defined with an ActorClass object
        as one of its base classes. To intentionally construct an ActorClass,
        use the '_ray_from_modified_class' classmethod.

        Raises:
            ActorClassInheritanceException: When ActorClass is inherited.
            AssertionError: If ActorClassInheritanceException is not raised i.e.,
                            conditions for raising it are not met in any
                            iteration of the loop.
            TypeError: In all other cases.
        """
        for base in bases:
            if isinstance(base, ActorClass):
                raise ActorClassInheritanceException(f"Attempted to define subclass '{name}' of actor class '{base.__ray_metadata__.class_name}'. Inheriting from actor classes is not currently supported. You can instead inherit from a non-actor base class and make the derived class an actor class (with @ray.remote).")
        assert False, 'ActorClass.__init__ should not be called. Please use the @ray.remote decorator instead.'

    def __call__(self, *args, **kwargs):
        """Prevents users from directly instantiating an ActorClass.

        This will be called instead of __init__ when 'ActorClass()' is executed
        because an is an object rather than a metaobject. To properly
        instantiated a remote actor, use 'ActorClass.remote()'.

        Raises:
            Exception: Always.
        """
        raise TypeError(f"Actors cannot be instantiated directly. Instead of '{self.__ray_metadata__.class_name}()', use '{self.__ray_metadata__.class_name}.remote()'.")

    @classmethod
    def _ray_from_modified_class(cls, modified_class, class_id, actor_options):
        for attribute in ['remote', '_remote', '_ray_from_modified_class', '_ray_from_function_descriptor']:
            if hasattr(modified_class, attribute):
                logger.warning(f'Creating an actor from class {modified_class.__name__} overwrites attribute {attribute} of that class')

        class DerivedActorClass(cls, modified_class):

            def __init__(self, *args, **kwargs):
                try:
                    cls.__init__(self, *args, **kwargs)
                except Exception as e:
                    if isinstance(e, TypeError) and (not isinstance(e, ActorClassInheritanceException)):
                        modified_class.__init__(self, *args, **kwargs)
                    else:
                        raise e
        name = f'ActorClass({modified_class.__name__})'
        DerivedActorClass.__module__ = modified_class.__module__
        DerivedActorClass.__name__ = name
        DerivedActorClass.__qualname__ = name
        self = DerivedActorClass.__new__(DerivedActorClass)
        actor_creation_function_descriptor = PythonFunctionDescriptor.from_class(modified_class.__ray_actor_class__)
        self.__ray_metadata__ = _ActorClassMetadata(Language.PYTHON, modified_class, actor_creation_function_descriptor, class_id, **_process_option_dict(actor_options))
        self._default_options = actor_options
        if 'runtime_env' in self._default_options:
            self._default_options['runtime_env'] = self.__ray_metadata__.runtime_env
        return self

    @classmethod
    def _ray_from_function_descriptor(cls, language, actor_creation_function_descriptor, actor_options):
        self = ActorClass.__new__(ActorClass)
        self.__ray_metadata__ = _ActorClassMetadata(language, None, actor_creation_function_descriptor, None, **_process_option_dict(actor_options))
        self._default_options = actor_options
        if 'runtime_env' in self._default_options:
            self._default_options['runtime_env'] = self.__ray_metadata__.runtime_env
        return self

    def remote(self, *args, **kwargs):
        """Create an actor.

        Args:
            args: These arguments are forwarded directly to the actor
                constructor.
            kwargs: These arguments are forwarded directly to the actor
                constructor.

        Returns:
            A handle to the newly created actor.
        """
        return self._remote(args=args, kwargs=kwargs, **self._default_options)

    def options(self, **actor_options):
        """Configures and overrides the actor instantiation parameters.

        The arguments are the same as those that can be passed
        to :obj:`ray.remote`.

        Args:
            num_cpus: The quantity of CPU cores to reserve
                for this task or for the lifetime of the actor.
            num_gpus: The quantity of GPUs to reserve
                for this task or for the lifetime of the actor.
            resources (Dict[str, float]): The quantity of various custom resources
                to reserve for this task or for the lifetime of the actor.
                This is a dictionary mapping strings (resource names) to floats.
            accelerator_type: If specified, requires that the task or actor run
                on a node with the specified type of accelerator.
                See :ref:`accelerator types <accelerator_types>`.
            memory: The heap memory request in bytes for this task/actor,
                rounded down to the nearest integer.
            object_store_memory: The object store memory request for actors only.
            max_restarts: This specifies the maximum
                number of times that the actor should be restarted when it dies
                unexpectedly. The minimum valid value is 0 (default),
                which indicates that the actor doesn't need to be restarted.
                A value of -1 indicates that an actor should be restarted
                indefinitely.
            max_task_retries: How many times to
                retry an actor task if the task fails due to a system error,
                e.g., the actor has died. If set to -1, the system will
                retry the failed task until the task succeeds, or the actor
                has reached its max_restarts limit. If set to `n > 0`, the
                system will retry the failed task up to n times, after which the
                task will throw a `RayActorError` exception upon :obj:`ray.get`.
                Note that Python exceptions are not considered system errors
                and don't trigger retries. [Internal use: You can override this number
                with the method's "_max_retries" option at @ray.method decorator or
                at .option() time.]
            max_pending_calls: Set the max number of pending calls
                allowed on the actor handle. When this value is exceeded,
                PendingCallsLimitExceeded will be raised for further tasks.
                Note that this limit is counted per handle. -1 means that the
                number of pending calls is unlimited.
            max_concurrency: The max number of concurrent calls to allow for
                this actor. This only works with direct actor calls. The max
                concurrency defaults to 1 for threaded execution, and 1000 for
                asyncio execution. Note that the execution order is not
                guaranteed when max_concurrency > 1.
            name: The globally unique name for the actor, which can be used
                to retrieve the actor via ray.get_actor(name) as long as the
                actor is still alive.
            namespace: Override the namespace to use for the actor. By default,
                actors are created in an anonymous namespace. The actor can
                be retrieved via ray.get_actor(name=name, namespace=namespace).
            lifetime: Either `None`, which defaults to the actor will fate
                share with its creator and will be deleted once its refcount
                drops to zero, or "detached", which means the actor will live
                as a global object independent of the creator.
            runtime_env (Dict[str, Any]): Specifies the runtime environment for
                this actor or task and its children. See
                :ref:`runtime-environments` for detailed documentation.
            scheduling_strategy: Strategy about how to
                schedule a remote function or actor. Possible values are
                None: ray will figure out the scheduling strategy to use, it
                will either be the PlacementGroupSchedulingStrategy using parent's
                placement group if parent has one and has
                placement_group_capture_child_tasks set to true,
                or "DEFAULT";
                "DEFAULT": default hybrid scheduling;
                "SPREAD": best effort spread scheduling;
                `PlacementGroupSchedulingStrategy`:
                placement group based scheduling;
                `NodeAffinitySchedulingStrategy`:
                node id based affinity scheduling.
            _metadata: Extended options for Ray libraries. For example,
                _metadata={"workflows.io/options": <workflow options>} for
                Ray workflows.

        Examples:

        .. code-block:: python

            @ray.remote(num_cpus=2, resources={"CustomResource": 1})
            class Foo:
                def method(self):
                    return 1
            # Class Bar will require 1 cpu instead of 2.
            # It will also require no custom resources.
            Bar = Foo.options(num_cpus=1, resources=None)
        """
        actor_cls = self
        default_options = self._default_options.copy()
        default_options.pop('concurrency_groups', None)
        updated_options = ray_option_utils.update_options(default_options, actor_options)
        ray_option_utils.validate_actor_options(updated_options, in_options=True)
        if 'runtime_env' in actor_options:
            updated_options['runtime_env'] = parse_runtime_env(updated_options['runtime_env'])

        class ActorOptionWrapper:

            def remote(self, *args, **kwargs):
                return actor_cls._remote(args=args, kwargs=kwargs, **updated_options)

            @DeveloperAPI
            def bind(self, *args, **kwargs):
                """
                For Ray DAG building that creates static graph from decorated
                class or functions.
                """
                from ray.dag.class_node import ClassNode
                return ClassNode(actor_cls.__ray_metadata__.modified_class, args, kwargs, updated_options)
        return ActorOptionWrapper()

    @wrap_auto_init
    @_tracing_actor_creation
    def _remote(self, args=None, kwargs=None, **actor_options):
        """Create an actor.

        This method allows more flexibility than the remote method because
        resource requirements can be specified and override the defaults in the
        decorator.

        Args:
            args: The arguments to forward to the actor constructor.
            kwargs: The keyword arguments to forward to the actor constructor.
            num_cpus: The number of CPUs required by the actor creation task.
            num_gpus: The number of GPUs required by the actor creation task.
            memory: Restrict the heap memory usage of this actor.
            resources: The custom resources required by the actor creation
                task.
            max_concurrency: The max number of concurrent calls to allow for
                this actor. This only works with direct actor calls. The max
                concurrency defaults to 1 for threaded execution, and 1000 for
                asyncio execution. Note that the execution order is not
                guaranteed when max_concurrency > 1.
            name: The globally unique name for the actor, which can be used
                to retrieve the actor via ray.get_actor(name) as long as the
                actor is still alive.
            namespace: Override the namespace to use for the actor. By default,
                actors are created in an anonymous namespace. The actor can
                be retrieved via ray.get_actor(name=name, namespace=namespace).
            lifetime: Either `None`, which defaults to the actor will fate
                share with its creator and will be deleted once its refcount
                drops to zero, or "detached", which means the actor will live
                as a global object independent of the creator.
            placement_group: (This has been deprecated, please use
                `PlacementGroupSchedulingStrategy` scheduling_strategy)
                the placement group this actor belongs to,
                or None if it doesn't belong to any group. Setting to "default"
                autodetects the placement group based on the current setting of
                placement_group_capture_child_tasks.
            placement_group_bundle_index: (This has been deprecated, please use
                `PlacementGroupSchedulingStrategy` scheduling_strategy)
                the index of the bundle
                if the actor belongs to a placement group, which may be -1 to
                specify any available bundle.
            placement_group_capture_child_tasks: (This has been deprecated,
                please use `PlacementGroupSchedulingStrategy`
                scheduling_strategy)
                Whether or not children tasks
                of this actor should implicitly use the same placement group
                as its parent. It is False by default.
            runtime_env (Dict[str, Any]): Specifies the runtime environment for
                this actor or task and its children (see
                :ref:`runtime-environments` for details).
            max_pending_calls: Set the max number of pending calls
                allowed on the actor handle. When this value is exceeded,
                PendingCallsLimitExceeded will be raised for further tasks.
                Note that this limit is counted per handle. -1 means that the
                number of pending calls is unlimited.
            scheduling_strategy: Strategy about how to schedule this actor.

        Returns:
            A handle to the newly created actor.
        """
        name = actor_options.get('name')
        namespace = actor_options.get('namespace')
        if name is not None:
            if not isinstance(name, str):
                raise TypeError(f"name must be None or a string, got: '{type(name)}'.")
            elif name == '':
                raise ValueError('Actor name cannot be an empty string.')
        if namespace is not None:
            ray._private.utils.validate_namespace(namespace)
        if actor_options.get('get_if_exists'):
            try:
                return ray.get_actor(name, namespace=namespace)
            except ValueError:
                updated_options = actor_options.copy()
                updated_options['get_if_exists'] = False
                try:
                    return self._remote(args, kwargs, **updated_options)
                except ValueError:
                    pass
                return ray.get_actor(name, namespace=namespace)
        actor_options.pop('concurrency_groups', None)
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        meta = self.__ray_metadata__
        actor_has_async_methods = len(inspect.getmembers(meta.modified_class, predicate=is_async_func)) > 0
        is_asyncio = actor_has_async_methods
        if actor_options.get('max_concurrency') is None:
            actor_options['max_concurrency'] = 1000 if is_asyncio else 1
        if client_mode_should_convert():
            return client_mode_convert_actor(self, args, kwargs, **actor_options)
        for k, v in ray_option_utils.actor_options.items():
            actor_options[k] = actor_options.get(k, v.default_value)
        actor_options.pop('concurrency_groups', None)
        max_concurrency = actor_options['max_concurrency']
        lifetime = actor_options['lifetime']
        runtime_env = actor_options['runtime_env']
        placement_group = actor_options['placement_group']
        placement_group_bundle_index = actor_options['placement_group_bundle_index']
        placement_group_capture_child_tasks = actor_options['placement_group_capture_child_tasks']
        scheduling_strategy = actor_options['scheduling_strategy']
        max_restarts = actor_options['max_restarts']
        max_task_retries = actor_options['max_task_retries']
        max_pending_calls = actor_options['max_pending_calls']
        if scheduling_strategy is None or not isinstance(scheduling_strategy, PlacementGroupSchedulingStrategy):
            _warn_if_using_deprecated_placement_group(actor_options, 3)
        worker = ray._private.worker.global_worker
        worker.check_connected()
        if name is not None:
            try:
                ray.get_actor(name, namespace=namespace)
            except ValueError:
                pass
            else:
                raise ValueError(f"The name {name} (namespace={namespace}) is already taken. Please use a different name or get the existing actor using ray.get_actor('{name}', namespace='{namespace}')")
        if lifetime is None:
            detached = None
        elif lifetime == 'detached':
            detached = True
        elif lifetime == 'non_detached':
            detached = False
        else:
            raise ValueError("actor `lifetime` argument must be one of 'detached', 'non_detached' and 'None'.")
        if worker.mode == ray.LOCAL_MODE:
            assert not meta.is_cross_language, 'Cross language ActorClass cannot be executed locally.'
        if not meta.is_cross_language and meta.last_export_session_and_job != worker.current_session_and_job:
            meta.last_export_session_and_job = worker.current_session_and_job
            worker.function_actor_manager.export_actor_class(meta.modified_class, meta.actor_creation_function_descriptor, meta.method_meta.methods.keys())
        resources = ray._private.utils.resources_from_ray_options(actor_options)
        if not set(resources.keys()).difference({'memory', 'object_store_memory'}):
            resources.setdefault('CPU', ray_constants.DEFAULT_ACTOR_CREATION_CPU_SIMPLE)
            actor_method_cpu = ray_constants.DEFAULT_ACTOR_METHOD_CPU_SIMPLE
        else:
            resources.setdefault('CPU', ray_constants.DEFAULT_ACTOR_CREATION_CPU_SPECIFIED)
            actor_method_cpu = ray_constants.DEFAULT_ACTOR_METHOD_CPU_SPECIFIED
        actor_placement_resources = {}
        assert actor_method_cpu in [0, 1]
        if actor_method_cpu == 1:
            actor_placement_resources = resources.copy()
            actor_placement_resources['CPU'] += 1
        if meta.is_cross_language:
            creation_args = cross_language._format_args(worker, args, kwargs)
        else:
            function_signature = meta.method_meta.signatures['__init__']
            creation_args = signature.flatten_args(function_signature, args, kwargs)
        if scheduling_strategy is None or isinstance(scheduling_strategy, PlacementGroupSchedulingStrategy):
            if isinstance(scheduling_strategy, PlacementGroupSchedulingStrategy):
                placement_group = scheduling_strategy.placement_group
                placement_group_bundle_index = scheduling_strategy.placement_group_bundle_index
                placement_group_capture_child_tasks = scheduling_strategy.placement_group_capture_child_tasks
            if placement_group_capture_child_tasks is None:
                placement_group_capture_child_tasks = worker.should_capture_child_tasks_in_placement_group
            placement_group = _configure_placement_group_based_on_context(placement_group_capture_child_tasks, placement_group_bundle_index, resources, actor_placement_resources, meta.class_name, placement_group=placement_group)
            if not placement_group.is_empty:
                scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group, placement_group_bundle_index, placement_group_capture_child_tasks)
            else:
                scheduling_strategy = 'DEFAULT'
        serialized_runtime_env_info = None
        if runtime_env is not None:
            serialized_runtime_env_info = get_runtime_env_info(runtime_env, is_job_runtime_env=False, serialize=True)
        concurrency_groups_dict = {}
        if meta.concurrency_groups is None:
            meta.concurrency_groups = []
        for cg_name in meta.concurrency_groups:
            concurrency_groups_dict[cg_name] = {'name': cg_name, 'max_concurrency': meta.concurrency_groups[cg_name], 'function_descriptors': []}
        for method_name in meta.method_meta.concurrency_group_for_methods:
            cg_name = meta.method_meta.concurrency_group_for_methods[method_name]
            assert cg_name in concurrency_groups_dict
            module_name = meta.actor_creation_function_descriptor.module_name
            class_name = meta.actor_creation_function_descriptor.class_name
            concurrency_groups_dict[cg_name]['function_descriptors'].append(PythonFunctionDescriptor(module_name, method_name, class_name))
        if meta.is_cross_language:
            func_name = '<init>'
            if meta.language == Language.CPP:
                func_name = meta.actor_creation_function_descriptor.function_name
            meta.actor_creation_function_descriptor = cross_language._get_function_descriptor_for_actor_method(meta.language, meta.actor_creation_function_descriptor, func_name, str(len(args) + len(kwargs)))
        actor_id = worker.core_worker.create_actor(meta.language, meta.actor_creation_function_descriptor, creation_args, max_restarts, max_task_retries, resources, actor_placement_resources, max_concurrency, detached, name if name is not None else '', namespace if namespace is not None else '', is_asyncio, extension_data=str(actor_method_cpu), serialized_runtime_env_info=serialized_runtime_env_info or '{}', concurrency_groups_dict=concurrency_groups_dict or dict(), max_pending_calls=max_pending_calls, scheduling_strategy=scheduling_strategy)
        if _actor_launch_hook:
            _actor_launch_hook(meta.actor_creation_function_descriptor, resources, scheduling_strategy)
        actor_handle = ActorHandle(meta.language, actor_id, max_task_retries, meta.method_meta.method_is_generator, meta.method_meta.decorators, meta.method_meta.signatures, meta.method_meta.num_returns, meta.method_meta.max_retries, meta.method_meta.retry_exceptions, meta.method_meta.generator_backpressure_num_objects, actor_method_cpu, meta.actor_creation_function_descriptor, worker.current_session_and_job, original_handle=True)
        return actor_handle

    @DeveloperAPI
    def bind(self, *args, **kwargs):
        """
        For Ray DAG building that creates static graph from decorated
        class or functions.
        """
        from ray.dag.class_node import ClassNode
        return ClassNode(self.__ray_metadata__.modified_class, args, kwargs, self._default_options)