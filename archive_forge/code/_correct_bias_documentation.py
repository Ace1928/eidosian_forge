import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.quantization
import torch.ao.ns._numeric_suite as ns
Compute the average of quantized and floating-point data from modules.

        The inputs x,y are output data from the quantized and floating-point modules.
        x is for the quantized module, y is for the floating point module
        