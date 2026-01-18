import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
def _make_standard_atom_mask() -> np.ndarray:
    """Returns [num_res_types, num_atom_types] mask array."""
    mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
    for restype, restype_letter in enumerate(restypes):
        restype_name = restype_1to3[restype_letter]
        atom_names = residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            mask[restype, atom_type] = 1
    return mask