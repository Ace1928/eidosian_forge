from abc import ABCMeta, abstractmethod
import enum
from typing import Optional, Union, Tuple
from scipy.sparse import spmatrix
import numpy as np
from pyomo.contrib.pynumero.sparse.base_block import BaseBlockVector, BaseBlockMatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
class LinearSolverStatus(enum.Enum):
    successful = 0
    not_enough_memory = 1
    singular = 2
    error = 3
    warning = 4
    max_iter = 5