from __future__ import annotations
from collections import OrderedDict
from typing import (
class ImmutableOrderedDict(immutabledict[_K, _V]):
    """
    An immutabledict subclass that maintains key order.

    Same as :class:`immutabledict` but based on :class:`collections.OrderedDict`.
    """
    dict_cls = OrderedDict