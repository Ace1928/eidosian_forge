import functools
import logging
from collections import abc
from typing import Union, Mapping, Any, Callable
from ray.rllib.core.models.specs.specs_base import Spec, TypeSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.nested_dict import NestedDict
from ray.util.annotations import DeveloperAPI
def _should_validate(cls_instance: object, method: Callable, tag: str='input') -> bool:
    """Returns True if the spec should be validated, False otherwise.

    The spec should be validated if the method is not cached (i.e. there is no cache
    storage attribute in the instance) or if the method is already cached. (i.e. it
    exists in the cache storage attribute)

    Args:
        cls_instance: The class instance that the method belongs to.
        method: The method to apply the spec checking to.
        tag: The tag of the spec to check. Either "input" or "output". This is used
        internally to defined an internal cache storage attribute based on the tag.

    Returns:
        True if the spec should be validated, False otherwise.
    """
    cache_store = getattr(cls_instance, f'__checked_{tag}_specs_cache__', None)
    return cache_store is None or method.__name__ not in cache_store