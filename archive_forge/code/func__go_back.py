from __future__ import annotations
import os
from fnmatch import fnmatch
from typing import (
import param
from ..io import PeriodicCallback
from ..layout import (
from ..util import fullpath
from ..viewable import Layoutable
from .base import CompositeWidget
from .button import Button
from .input import TextInput
from .select import CrossSelector
def _go_back(self, event: param.parameterized.Event):
    self._position -= 1
    self._directory.value = self._stack[self._position]
    self._update_files()
    self._forward.disabled = False
    if self._position == 0:
        self._back.disabled = True