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