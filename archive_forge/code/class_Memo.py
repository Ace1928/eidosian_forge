import sys
import math
from datetime import datetime
from sentry_sdk.utils import (
from sentry_sdk._compat import (
from sentry_sdk._types import TYPE_CHECKING
class Memo(object):
    __slots__ = ('_ids', '_objs')

    def __init__(self):
        self._ids = {}
        self._objs = []

    def memoize(self, obj):
        self._objs.append(obj)
        return self

    def __enter__(self):
        obj = self._objs[-1]
        if id(obj) in self._ids:
            return True
        else:
            self._ids[id(obj)] = obj
            return False

    def __exit__(self, ty, value, tb):
        self._ids.pop(id(self._objs.pop()), None)