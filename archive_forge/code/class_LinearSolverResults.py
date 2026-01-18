from abc import ABCMeta, abstractmethod
import enum
from typing import Optional, Union, Tuple
from scipy.sparse import spmatrix
import numpy as np
from pyomo.contrib.pynumero.sparse.base_block import BaseBlockVector, BaseBlockMatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
class LinearSolverResults(object):

    def __init__(self, status: Optional[LinearSolverStatus]=None):
        self.status = status