import collections
import os
import typing
from dataclasses import dataclass
def _flexible_repr(self):
    return f'{self.__class__.__qualname__}(' + ', '.join((f'{key}={val!r}' for key, val in self.__dict__.items())) + ')'