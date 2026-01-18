from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
class _ArgFormat(object):

    def __init__(self, func, ignore_none=None, ignore_missing_value=False):
        self.func = func
        self.ignore_none = ignore_none
        self.ignore_missing_value = ignore_missing_value

    def __call__(self, value, ctx_ignore_none):
        ignore_none = self.ignore_none if self.ignore_none is not None else ctx_ignore_none
        if value is None and ignore_none:
            return []
        f = self.func
        return [str(x) for x in f(value)]