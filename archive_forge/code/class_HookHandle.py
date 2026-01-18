import os
import sys
import hashlib
import uuid
import warnings
import collections
import weakref
import requests
import numpy as np
from .. import ndarray
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
class HookHandle(object):
    """A handle that can attach/detach a hook."""

    def __init__(self):
        self._hooks_dict_ref = None
        self._id = None

    def attach(self, hooks_dict, hook):
        assert not self._hooks_dict_ref, 'The same handle cannot be attached twice.'
        self._id = id(hook)
        hooks_dict[self._id] = hook
        self._hooks_dict_ref = weakref.ref(hooks_dict)

    def detach(self):
        hooks_dict = self._hooks_dict_ref()
        if hooks_dict is not None and self._id in hooks_dict:
            del hooks_dict[self._id]

    def __getstate__(self):
        return (self._hooks_dict_ref(), self._id)

    def __setstate__(self, state):
        if state[0] is None:
            self._hooks_dict_ref = weakref.ref(collections.OrderedDict())
        else:
            self._hooks_dict_ref = weakref.ref(state[0])
        self._id = state[1]

    def __enter__(self):
        return self

    def __exit__(self, ptype, value, trace):
        self.detach()