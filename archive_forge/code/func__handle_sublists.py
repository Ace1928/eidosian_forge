from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Sequence
from ..models.graphs import StaticLayoutProvider
from ..models.renderers import GraphRenderer
from ..util.warnings import warn
def _handle_sublists(values):
    if any((isinstance(x, (list, tuple)) for x in values)):
        if not all((isinstance(x, (list, tuple)) for x in values if x is not None)):
            raise ValueError("Can't mix scalar and non-scalar values for graph attributes")
        return [[] if x is None else list(x) for x in values]
    return values