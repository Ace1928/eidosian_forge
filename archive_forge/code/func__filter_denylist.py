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
def _filter_denylist(self, event: param.parameterized.Event):
    """
        Ensure that if unselecting a currently selected path and it
        is not in the current working directory then it is removed
        from the denylist.
        """
    dirs, files = _scan_path(self._cwd, self.file_pattern)
    paths = [('📁' if p in dirs else '') + os.path.relpath(p, self._cwd) for p in dirs + files]
    denylist = self._selector._lists[False]
    options = dict(self._selector._items)
    self._selector.options.clear()
    self._selector.options.update([(k, v) for k, v in options.items() if k in paths or v in self.value])
    denylist.options = [o for o in denylist.options if o in paths]