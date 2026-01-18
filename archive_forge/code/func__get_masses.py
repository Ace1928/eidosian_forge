import io
import re
import shlex
import warnings
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any, Sequence
import collections.abc
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data, Spacegroup
from ase.io.cif_unicode import format_unicode, handle_subscripts
from ase.utils import iofunction
def _get_masses(self) -> Optional[np.ndarray]:
    mask = self._where_deuterium()
    if not any(mask):
        return None
    symbols = self.get_symbols()
    masses = Atoms(symbols).get_masses()
    masses[mask] = 2.01355
    return masses