import functools
import pathlib
from contextlib import suppress
from types import SimpleNamespace
from .. import readers, _adapters
def _file_reader(self):
    try:
        path = pathlib.Path(self.spec.origin)
    except TypeError:
        return None
    if path.exists():
        return readers.FileReader(SimpleNamespace(path=path))