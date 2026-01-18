from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json, encode
from ase.utils import opencew
def _dump_json(self):
    target = self._filename
    if target.exists():
        raise RuntimeError(f'Already exists: {target}')
    self.directory.mkdir(exist_ok=True, parents=True)
    write_json(target, self._dct)