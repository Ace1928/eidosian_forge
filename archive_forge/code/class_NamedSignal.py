from __future__ import annotations
import typing as t
from collections import defaultdict
from contextlib import contextmanager
from inspect import iscoroutinefunction
from warnings import warn
from weakref import WeakValueDictionary
from blinker._utilities import annotatable_weakref
from blinker._utilities import hashable_identity
from blinker._utilities import IdentityType
from blinker._utilities import lazy_property
from blinker._utilities import reference
from blinker._utilities import symbol
from blinker._utilities import WeakTypes
class NamedSignal(Signal):
    """A named generic notification emitter."""

    def __init__(self, name: str, doc: str | None=None) -> None:
        Signal.__init__(self, doc)
        self.name = name

    def __repr__(self) -> str:
        base = Signal.__repr__(self)
        return f'{base[:-1]}; {self.name!r}>'