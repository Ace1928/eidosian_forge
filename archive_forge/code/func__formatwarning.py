from __future__ import annotations
import logging # isort:skip
import importlib.metadata as importlib_metadata
from . import sampledata # isort:skip
from .util import logconfig # isort:skip
import warnings # isort:skip
from .util.warnings import BokehDeprecationWarning, BokehUserWarning # isort:skip
def _formatwarning(message, category, filename, lineno, line=None):
    from .util.warnings import BokehDeprecationWarning, BokehUserWarning
    if category not in (BokehDeprecationWarning, BokehUserWarning):
        return original_formatwarning(message, category, filename, lineno, line)
    return f'{category.__name__}: {message}\n'