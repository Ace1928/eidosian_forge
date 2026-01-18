import dataclasses
import re
import string
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import residue_constants
@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""
    atom_positions: np.ndarray
    aatype: np.ndarray
    atom_mask: np.ndarray
    residue_index: np.ndarray
    b_factors: np.ndarray
    chain_index: Optional[np.ndarray] = None
    remark: Optional[str] = None
    parents: Optional[Sequence[str]] = None
    parents_chain_index: Optional[Sequence[int]] = None