from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
def _get_filter_stack_for(preferences: Preferences) -> Stack:
    """Get the filter stack given a sequence of preferences.

    Args:
      preferences: a sequence of (name,value) tuples where
          name is the preference name and
          value is the key into the filter stack map registered
          for that preference.
    """
    if preferences is None:
        return []
    stack = _stack_cache.get(preferences)
    if stack is not None:
        return stack
    stack = []
    for k, v in preferences:
        if v is None:
            continue
        try:
            stack_map_lookup = filter_stacks_registry.get(k)
        except KeyError:
            continue
        items = stack_map_lookup(v)
        if items:
            stack.extend(items)
    _stack_cache[preferences] = stack
    return stack