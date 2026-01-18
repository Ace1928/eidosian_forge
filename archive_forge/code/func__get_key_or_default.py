from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
from .pyutils import get_named_object
def _get_key_or_default(self, key=None):
    """Return either 'key' or the default key if key is None"""
    if key is not None:
        return key
    if self.default_key is None:
        raise KeyError('Key is None, and no default key is set')
    else:
        return self.default_key