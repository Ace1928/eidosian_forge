from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from types import FunctionType, MethodType
from typing import Generic, TypeVar, Optional, List
def _wrap_reset(m):

    @wraps(m)
    def w(self, *args, **kwargs):
        p = self.index
        try:
            return m(self, *args, **kwargs)
        except nomatch:
            self.index = p
            raise
    return w