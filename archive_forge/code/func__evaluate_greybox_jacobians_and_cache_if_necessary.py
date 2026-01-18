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
def _evaluate_greybox_jacobians_and_cache_if_necessary(self):
    if self._greybox_jac_cached:
        return
    jac = BlockMatrix(len(self._external_greybox_helpers), 1)
    for i, external in enumerate(self._external_greybox_helpers):
        jac.set_block(i, 0, external.evaluate_jacobian())
    self._cached_greybox_jac = jac.tocoo()
    self._greybox_jac_cached = True