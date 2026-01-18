import sys
import warnings
import torch
from torch.onnx import symbolic_opset11 as opset11
from torch.onnx.symbolic_helper import parse_args
def _process_rois_for_roi_align(g, rois):
    return opset11.select(g, rois, 1, g.op('Constant', value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))