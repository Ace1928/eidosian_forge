from __future__ import annotations
from . import Image
def __dispatcher(self, action, *args):
    return getattr(self, 'ui_handle_' + action)(*args)