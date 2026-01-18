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
def _get_sitesym(self):
    sitesym = self._get_any(['_space_group_symop_operation_xyz', '_space_group_symop.operation_xyz', '_symmetry_equiv_pos_as_xyz'])
    if isinstance(sitesym, str):
        sitesym = [sitesym]
    return sitesym