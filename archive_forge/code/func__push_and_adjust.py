import json
from typing import IO, Any, Tuple, List
from .parser import Parser
from .symbols import (
def _push_and_adjust(self, symbol=None):
    self._push()
    if isinstance(self._current, dict) and self._key is not None:
        if self._key not in self._current:
            self._current = symbol.get_default()
        else:
            self._current = self._current[self._key]