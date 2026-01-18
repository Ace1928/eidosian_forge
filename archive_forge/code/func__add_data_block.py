from pyomo.environ import (
from pyomo.common.sorting import sorted_robust
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface, InTempDir
import logging
import os
import io
import shutil
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy, scipy_available
def _add_data_block(self, existing_block=None):
    if existing_block is not None:
        if hasattr(existing_block, '_has_replaced_expressions') and (not existing_block._has_replaced_expressions):
            for var, _, _, _ in existing_block._sens_data_list:
                var.fix()
            self.model_instance.del_component(existing_block)
        else:
            msg = 'Re-using sensitivity interface is not supported when calculating sensitivity for mutable parameters. Used fixed vars instead if you want to do this.'
            raise RuntimeError(msg)
    block = Block()
    self.model_instance.add_component(self.get_default_block_name(), block)
    self.block = block
    block._has_replaced_expressions = False
    block._sens_data_list = []
    block._paramList = None
    block.constList = ConstraintList()
    return block