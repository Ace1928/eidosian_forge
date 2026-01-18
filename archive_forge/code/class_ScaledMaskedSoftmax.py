import torch
from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType
from fused_softmax_lib import (
class ScaledMaskedSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_masked_softmax_forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_masked_softmax_backward(output_grads, softmax_results, scale_t[0])
        return (input_grads, None, None)