from typing import Optional
import numpy as np
from onnx.reference.ops._op import OpRun
class Slice_10(SliceCommon):

    def __init__(self, onnx_node, run_params):
        SliceCommon.__init__(self, onnx_node, run_params)