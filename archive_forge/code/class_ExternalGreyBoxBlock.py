import abc
import logging
from scipy.sparse import coo_matrix
from pyomo.common.dependencies import numpy as np
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base import Var, Set, Constraint, value
from pyomo.core.base.block import _BlockData, Block, declare_custom_block
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.set import UnindexedComponent_set
from pyomo.core.base.reference import Reference
from ..sparse.block_matrix import BlockMatrix
class ExternalGreyBoxBlock(Block):
    _ComponentDataClass = ExternalGreyBoxBlockData

    def __new__(cls, *args, **kwds):
        if cls != ExternalGreyBoxBlock:
            target_cls = cls
        elif not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            target_cls = ScalarExternalGreyBoxBlock
        else:
            target_cls = IndexedExternalGreyBoxBlock
        return super(ExternalGreyBoxBlock, cls).__new__(target_cls)

    def __init__(self, *args, **kwds):
        kwds.setdefault('ctype', ExternalGreyBoxBlock)
        self._init_model = Initializer(kwds.pop('external_model', None))
        Block.__init__(self, *args, **kwds)

    def construct(self, data=None):
        """
        Construct the ExternalGreyBoxBlockDatas
        """
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug('Constructing external grey box model %s' % self.name)
        super(ExternalGreyBoxBlock, self).construct(data)
        if self._init_model is not None:
            block = self.parent_block()
            for index, data in self.items():
                data.set_external_model(self._init_model(block, index))