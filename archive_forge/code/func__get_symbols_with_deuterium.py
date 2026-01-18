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
def _get_symbols_with_deuterium(self):
    labels = self._get_any(['_atom_site_type_symbol', '_atom_site_label'])
    if labels is None:
        raise NoStructureData('No symbols')
    symbols = []
    for label in labels:
        if label == '.' or label == '?':
            raise NoStructureData('Symbols are undetermined')
        match = re.search('([A-Z][a-z]?)', label)
        symbol = match.group(0)
        symbols.append(symbol)
    return symbols