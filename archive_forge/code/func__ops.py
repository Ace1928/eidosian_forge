from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
@property
def _ops(self):
    return tuple(map(self._get_operation, self.patch))