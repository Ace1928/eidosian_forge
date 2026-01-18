import os
import numpy as np
import logging
from scipy.sparse import coo_matrix, identity
from pyomo.common.deprecation import deprecated
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.utils import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedNLP
def _evaluate_hessian_if_necessary_and_cache(self):
    if self._cached_hessian is None:
        hess = BlockMatrix(2, 2)
        hess.set_row_size(0, self._ex_model.n_inputs())
        hess.set_row_size(1, self._ex_model.n_outputs())
        hess.set_col_size(0, self._ex_model.n_inputs())
        hess.set_col_size(1, self._ex_model.n_outputs())
        eq_hess = None
        if self._ex_model.n_equality_constraints() > 0:
            eq_hess = self._ex_model.evaluate_hessian_equality_constraints()
            if np.any(eq_hess.row < eq_hess.col):
                raise ValueError('ExternalGreyBoxModel must return lower triangular portion of the Hessian only')
            eq_hess = make_lower_triangular_full(eq_hess)
        output_hess = None
        if self._ex_model.n_outputs() > 0:
            output_hess = self._ex_model.evaluate_hessian_outputs()
            if np.any(output_hess.row < output_hess.col):
                raise ValueError('ExternalGreyBoxModel must return lower triangular portion of the Hessian only')
            output_hess = make_lower_triangular_full(output_hess)
        input_hess = None
        if eq_hess is not None and output_hess is not None:
            row = np.concatenate((eq_hess.row, output_hess.row))
            col = np.concatenate((eq_hess.col, output_hess.col))
            data = np.concatenate((eq_hess.data, output_hess.data))
            assert eq_hess.shape == output_hess.shape
            input_hess = coo_matrix((data, (row, col)), shape=eq_hess.shape)
        elif eq_hess is not None:
            input_hess = eq_hess
        elif output_hess is not None:
            input_hess = output_hess
        assert input_hess is not None
        hess.set_block(0, 0, input_hess)
        self._cached_hessian = hess.tocoo()