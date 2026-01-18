from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
class MetaSuper(type):
    """adding .__super"""

    def __init__(cls, name: str, bases, d):
        super().__init__(name, bases, d)
        if hasattr(cls, f'_{name}__super'):
            raise AttributeError('Class has same name as one of its super classes')

        @property
        def _super(self):
            warnings.warn(f'`{name}.__super` was a deprecated feature for old python versions.Please use `super()` call instead.', DeprecationWarning, stacklevel=3)
            return super(cls, self)
        setattr(cls, f'_{name}__super', _super)