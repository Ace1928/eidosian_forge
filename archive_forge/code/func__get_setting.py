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
def _get_setting(self) -> Optional[int]:
    setting_str = self.get('_symmetry_space_group_setting')
    if setting_str is None:
        return None
    setting = int(setting_str)
    if setting not in [1, 2]:
        raise ValueError(f'Spacegroup setting must be 1 or 2, not {setting}')
    return setting