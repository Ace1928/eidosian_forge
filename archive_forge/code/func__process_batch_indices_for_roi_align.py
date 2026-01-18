import sys
import warnings
import torch
from torch.onnx import symbolic_opset11 as opset11
from torch.onnx.symbolic_helper import parse_args
def _process_batch_indices_for_roi_align(g, rois):
    indices = opset11.squeeze(g, opset11.select(g, rois, 1, g.op('Constant', value_t=torch.tensor([0], dtype=torch.long))), 1)
    return g.op('Cast', indices, to_i=torch.onnx.TensorProtoDataType.INT64)