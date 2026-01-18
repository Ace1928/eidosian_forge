import torch
from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType
from fused_softmax_lib import (
class FusedScaleMaskSoftmax(torch.nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(self, input_in_fp16, input_in_bf16, attn_mask_type, scaled_masked_softmax_fusion, mask_func, softmax_in_fp32, scale):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        if self.input_in_fp16 and self.input_in_bf16:
            raise RuntimeError('both fp16 and bf16 flags cannot be active at the same time.')
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        if not (self.scale is None or softmax_in_fp32):
            raise RuntimeError('softmax should be in fp32 when scaled')
        if self.scaled_masked_softmax_fusion:
            if self.attn_mask_type == AttnMaskType.causal:
                self.fused_softmax_func = scaled_upper_triang_masked_softmax
            elif self.attn_mask_type == AttnMaskType.padding:
                self.fused_softmax_func = scaled_masked_softmax
            else:
                raise ValueError('Invalid attn_mask_type.')

    def forward(self, input, mask):
        assert input.dim() == 4
        if self.is_kernel_available(mask, *input.size()):
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np
        if self.scaled_masked_softmax_fusion and self.input_in_float16 and (self.attn_mask_type == AttnMaskType.causal or (self.attn_mask_type == AttnMaskType.padding and mask is not None)) and (16 < sk <= 8192) and (sq % 4 == 0) and (sk % 4 == 0) and (attn_batches % 4 == 0):
            if 0 <= sk <= 8192:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)
                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                elif sq % batch_per_block == 0:
                    return True
        return False

    def forward_fused_softmax(self, input, mask):
        scale = self.scale if self.scale is not None else 1.0
        return self.fused_softmax_func(input, mask, scale)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()
        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)
        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()
        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        return scaled_masked_softmax_get_batch_per_block(sq, sk, b, np)