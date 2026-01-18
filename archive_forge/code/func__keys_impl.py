from __future__ import annotations
from collections.abc import MutableSet
from copy import deepcopy
from .. import exceptions
from .._internal import _missing
from .mixins import ImmutableDictMixin
from .mixins import ImmutableListMixin
from .mixins import ImmutableMultiDictMixin
from .mixins import UpdateDictMixin
from .. import http
def _keys_impl(self):
    """This function exists so __len__ can be implemented more efficiently,
        saving one list creation from an iterator.
        """
    rv = set()
    rv.update(*self.dicts)
    return rv