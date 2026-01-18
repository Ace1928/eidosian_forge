from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
from .pyutils import get_named_object
def _set_default_key(self, key):
    if key not in self._dict:
        raise KeyError('No object registered under key %s.' % key)
    else:
        self._default_key = key