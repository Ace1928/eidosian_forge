import collections
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer
def _read_everything(self):
    yield from self._read_energy()
    with (self.path / 'INFO.OUT').open() as fd:
        yield from parse_elk_info(fd)
    with (self.path / 'EIGVAL.OUT').open() as fd:
        yield from parse_elk_eigval(fd)
    with (self.path / 'KPOINTS.OUT').open() as fd:
        yield from parse_elk_kpoints(fd)