import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
def _should_send_property(self, key, value):
    """Check the property lock (property_lock)"""
    to_json = self.trait_metadata(key, 'to_json', self._trait_to_json)
    if key in self._property_lock:
        split_value = _remove_buffers({key: to_json(value, self)})
        split_lock = _remove_buffers({key: self._property_lock[key]})
        if jsonloads(jsondumps(split_value[0])) == split_lock[0] and split_value[1] == split_lock[1] and _buffer_list_equal(split_value[2], split_lock[2]):
            if self._holding_sync:
                self._states_to_send.discard(key)
            return False
    if self._holding_sync:
        self._states_to_send.add(key)
        return False
    else:
        return True