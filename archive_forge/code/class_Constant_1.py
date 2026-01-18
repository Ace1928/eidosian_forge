import numpy as np
from onnx.reference.custom_element_types import (
from onnx.reference.op_run import OpRun, RefAttrName
class Constant_1(ConstantCommon):

    def __init__(self, onnx_node, run_params):
        ConstantCommon.__init__(self, onnx_node, run_params)
        self.cst = self.value
        _check_dtype(self.cst)

    def _run(self, **overridden_attributes):
        if overridden_attributes and (len(overridden_attributes) > 1 or 'value' not in overridden_attributes or id(overridden_attributes['value']) != id(self.value)):
            raise RuntimeError('Function attributes are not implemented for opset <= 11. Use opset > 12.')
        return (self._check(self.cst),)