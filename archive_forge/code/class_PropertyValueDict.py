from __future__ import annotations
import logging # isort:skip
import copy
from typing import (
import numpy as np
class PropertyValueDict(PropertyValueContainer, dict):
    """ A dict property value container that supports change notifications on
    mutating operations.

    When a Bokeh model has a ``List`` property, the ``PropertyValueLists`` are
    transparently created to wrap those values. These ``PropertyValueList``
    values are subject to normal property validation. If the property type
    ``foo = Dict(Str, Str)`` then attempting to set ``x.foo['bar'] = 10`` will
    raise an error.

    Instances of ``PropertyValueDict`` can be eplicitly created by passing
    any object that the standard dict initializer accepts, for example:

    .. code-block:: python

        >>> PropertyValueDict(dict(a=10, b=20))
        {'a': 10, 'b': 20}

        >>> PropertyValueDict(a=10, b=20)
        {'a': 10, 'b': 20}

        >>> PropertyValueDict([('a', 10), ['b', 20]])
        {'a': 10, 'b': 20}

    The following mutating operations on dicts automatically trigger
    notifications:

    .. code-block:: python

        del x[y]
        x[i] = y
        x.clear
        x.pop
        x.popitem
        x.setdefault
        x.update

    """

    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)

    def _saved_copy(self):
        return dict(self)

    @notify_owner
    def __delitem__(self, y):
        return super().__delitem__(y)

    @notify_owner
    def __setitem__(self, i, y):
        return super().__setitem__(i, y)

    @notify_owner
    def clear(self):
        return super().clear()

    @notify_owner
    def pop(self, *args):
        return super().pop(*args)

    @notify_owner
    def popitem(self):
        return super().popitem()

    @notify_owner
    def setdefault(self, *args):
        return super().setdefault(*args)

    @notify_owner
    def update(self, *args, **kwargs):
        return super().update(*args, **kwargs)