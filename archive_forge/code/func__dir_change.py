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
def _dir_change(self, event: param.parameterized.Event):
    path = fullpath(self._directory.value)
    if not path.startswith(self._root_directory):
        self._directory.value = self._root_directory
        return
    elif path != self._directory.value:
        self._directory.value = path
    self._go.disabled = path == self._cwd