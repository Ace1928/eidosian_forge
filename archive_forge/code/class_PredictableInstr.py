from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class PredictableInstr(Instruction):

    def set_weights(self, weights):
        operands = [MetaDataString(self.module, 'branch_weights')]
        for w in weights:
            if w < 0:
                raise ValueError('branch weight must be a positive integer')
            operands.append(Constant(types.IntType(32), w))
        md = self.module.add_metadata(operands)
        self.set_metadata('prof', md)