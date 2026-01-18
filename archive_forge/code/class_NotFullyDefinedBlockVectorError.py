import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector
class NotFullyDefinedBlockVectorError(Exception):
    pass