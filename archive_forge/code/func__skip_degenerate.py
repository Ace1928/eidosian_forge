import functools
import pathlib
from contextlib import suppress
from types import SimpleNamespace
from .. import readers, _adapters
def _skip_degenerate(reader):
    """
    Mask any degenerate reader. Ref #298.
    """
    is_degenerate = isinstance(reader, _adapters.CompatibilityFiles) and (not reader._reader)
    return reader if not is_degenerate else None