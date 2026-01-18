import logging
import types
import weakref
from dataclasses import dataclass
from . import config
def _get_weakref_from_f_locals(frame: types.FrameType, local_name: str):
    obj = frame.f_locals.get(local_name, None)
    weak_id = None
    try:
        weak_id = weakref.ref(obj)
    except TypeError:
        pass
    return weak_id