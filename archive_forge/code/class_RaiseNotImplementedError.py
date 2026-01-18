from __future__ import annotations
from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
import itertools as itl
class RaiseNotImplementedError:
    """Raise ``NotImplementedError`` when called."""

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def __call__(self, *args, **kwargs):
        types = tuple((type(a) for a in args))
        raise NotImplementedError('Ambiguous signature for %s: <%s>' % (self.dispatcher.name, str_signature(types)))