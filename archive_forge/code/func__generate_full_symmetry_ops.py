from __future__ import annotations
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Literal, overload
import numpy as np
from monty.design_patterns import cached_class
from monty.serialization import loadfn
from pymatgen.util.string import Stringify
def _generate_full_symmetry_ops(self) -> np.ndarray:
    symm_ops = np.array(self.generators)
    for op in symm_ops:
        op[0:3, 3] = np.mod(op[0:3, 3], 1)
    new_ops = symm_ops
    while len(new_ops) > 0 and len(symm_ops) < self.order:
        gen_ops = []
        for g in new_ops:
            temp_ops = np.einsum('ijk,kl', symm_ops, g)
            for op in temp_ops:
                op[0:3, 3] = np.mod(op[0:3, 3], 1)
                ind = np.where(np.abs(1 - op[0:3, 3]) < 1e-05)
                op[ind, 3] = 0
                if not in_array_list(symm_ops, op):
                    gen_ops.append(op)
                    symm_ops = np.append(symm_ops, [op], axis=0)
        new_ops = gen_ops
    assert len(symm_ops) == self.order
    return symm_ops