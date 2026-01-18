import os
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import WriterFactory
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.common.env import CtypesEnviron
from ..sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from .external_grey_box import ExternalGreyBoxBlock
def _evaluate_greybox_hessians_and_cache_if_necessary(self):
    if self._greybox_hess_cached:
        return
    data = list()
    irow = list()
    jcol = list()
    for external in self._external_greybox_helpers:
        hess = external.evaluate_hessian()
        data.append(hess.data)
        irow.append(hess.row)
        jcol.append(hess.col)
    data = np.concatenate(data)
    irow = np.concatenate(irow)
    jcol = np.concatenate(jcol)
    self._cached_greybox_hess = coo_matrix((data, (irow, jcol)), shape=(self.n_primals(), self.n_primals()))
    self._greybox_hess_cached = True