import sys
import warnings
import torch
from torch.onnx import symbolic_opset11 as opset11
from torch.onnx.symbolic_helper import parse_args
def _process_sampling_ratio_for_roi_align(g, sampling_ratio: int):
    if sampling_ratio < 0:
        warnings.warn('ONNX export for RoIAlign with a non-zero sampling_ratio is not supported. The model will be exported with a sampling_ratio of 0.')
        sampling_ratio = 0
    return sampling_ratio