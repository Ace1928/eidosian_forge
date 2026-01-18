from __future__ import annotations
import logging
import re
from itertools import chain, combinations
from typing import TYPE_CHECKING, no_type_check, overload
import numpy as np
from monty.fractions import gcd_float
from monty.json import MontyDecoder, MSONable
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
def _balance_coeffs(self, comp_matrix, max_num_constraints):
    first_product_idx = len(self._input_reactants)
    product_constraints = chain.from_iterable([combinations(range(first_product_idx, self._num_comp), n_constr) for n_constr in range(max_num_constraints, 0, -1)])
    reactant_constraints = chain.from_iterable([combinations(range(first_product_idx), n_constr) for n_constr in range(max_num_constraints, 0, -1)])
    best_soln = None
    balanced = False
    for constraints in chain(product_constraints, reactant_constraints):
        n_constr = len(constraints)
        comp_and_constraints = np.append(comp_matrix, np.zeros((n_constr, self._num_comp)), axis=0)
        b = np.zeros((self._num_elems + n_constr, 1))
        b[-n_constr:] = 1 if min(constraints) >= first_product_idx else -1
        for num, idx in enumerate(constraints):
            comp_and_constraints[self._num_elems + num, idx] = 1
        coeffs = np.matmul(np.linalg.pinv(comp_and_constraints), b)
        if np.allclose(np.matmul(comp_matrix, coeffs), np.zeros((self._num_elems, 1))):
            balanced = True
            expected_signs = np.array([-1] * len(self._input_reactants) + [+1] * len(self._input_products))
            num_errors = np.sum(np.multiply(expected_signs, coeffs.T) < self.TOLERANCE)
            if num_errors == 0:
                self._lowest_num_errors = 0
                return np.squeeze(coeffs)
            if num_errors < self._lowest_num_errors:
                self._lowest_num_errors = num_errors
                best_soln = coeffs
    if not balanced:
        raise ReactionError('Reaction cannot be balanced.')
    return np.squeeze(best_soln)