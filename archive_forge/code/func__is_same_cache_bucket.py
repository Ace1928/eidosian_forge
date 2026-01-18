import logging
import types
import weakref
from dataclasses import dataclass
from . import config
def _is_same_cache_bucket(frame: types.FrameType, cache_entry) -> bool:
    """
    Checks if the ID_MATCH'd objects saved on cache_entry are same as the ones
    in frame.f_locals, and if the config hash used to compile the cache entry's
    optimized code is the same as the frame's.
    """
    from .eval_frame import get_saved_else_current_config_hash
    if not cache_entry:
        return False
    if cache_entry.check_fn.config_hash != get_saved_else_current_config_hash():
        return False
    for local_name, weakref_from_cache_entry in cache_entry.check_fn.id_matched_objs.items():
        if weakref_from_cache_entry() is not None:
            weakref_from_frame = _get_weakref_from_f_locals(frame, local_name)
            if weakref_from_frame != weakref_from_cache_entry:
                return False
    return True