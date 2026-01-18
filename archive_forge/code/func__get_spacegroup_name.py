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
def _get_spacegroup_name(self):
    hm_symbol = self._get_any(['_space_group_name_h-m_alt', '_symmetry_space_group_name_h-m', '_space_group.Patterson_name_h-m', '_space_group.patterson_name_h-m'])
    hm_symbol = old_spacegroup_names.get(hm_symbol, hm_symbol)
    return hm_symbol