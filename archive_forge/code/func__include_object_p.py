from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
def _include_object_p(self, obj):
    if obj is self:
        return False
    kind = type(obj)
    if kind == type(self._include_object_p):
        try:
            exact_method_equals = self._include_object_p.__eq__(obj)
        except AttributeError:
            exact_method_equals = self._include_object_p.__cmp__(obj) == 0
        if exact_method_equals is not NotImplemented and exact_method_equals:
            return False
    for x in self.__dict__.values():
        if obj is x:
            return False
    if kind in self.ignored_types or kind in self.IGNORED_TYPES:
        return False
    return True