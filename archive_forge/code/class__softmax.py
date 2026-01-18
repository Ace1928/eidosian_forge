import torch
from ... import jit
from ... import language as tl
from ... import next_power_of_2
class _softmax(torch.autograd.Function):

    @staticmethod
    def make_lut(layout, block, device):
        _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
        sizes = _empty.clone()
        for h in range(layout.shape[0]):
            sizes = torch.cat((sizes, layout[h, :, :].sum(-1)))
        total_sizes = sizes * block
        offsets = torch.zeros_like(sizes)
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
        columns = layout.nonzero(as_tuple=False)[:, 2]
        header = torch.stack((sizes, offsets), dim=1).view(-1)
        lut = torch.cat((header, columns)).type(torch.int32).to(device)
        return (lut, int(total_sizes.max()))

    @staticmethod
    def forward(ctx, a, scale, rel_logits, is_causal, spdims, block, lut, maxlut, is_dense):
        if scale is not None and isinstance(scale, torch.Tensor):
            assert scale.device.type == 'cpu'
            scale = scale.item()
        M = a.shape[0]
        grid = [spdims[0], spdims[1] * block, M]
        rel_shape = (1, 1, 1, 1) if rel_logits is None else rel_logits.shape
        rel_strides = (1, 1, 1, 1) if rel_logits is None else rel_logits.stride()
        out = torch.empty_like(a)
        _blocksparse_softmax_fwd[grid](out, a, a.stride(0), lut, rel_logits, rel_shape[-1], rel_strides[0], rel_strides[1], scale, is_causal, BLOCK_SIZE=block, ROW_SIZE=next_power_of_2(maxlut), IS_DENSE=is_dense, num_warps=num_warps(maxlut))
        ctx.save_for_backward(out, lut)
        ctx.spdims = spdims
        ctx.block = block
        ctx.maxlut = maxlut
        ctx.scale = scale
        ctx.rel_shape = rel_shape
        ctx.rel_strides = rel_strides
        ctx.rel_dtype = a.dtype
        ctx.is_dense = is_dense
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout):
        out, lut = ctx.saved_tensors
        dr = None
        if ctx.needs_input_grad[3]:
            dr = torch.zeros(ctx.rel_shape, dtype=ctx.rel_dtype, device=out.device)
        M = out.shape[0]
        grid = (ctx.spdims[0], ctx.spdims[1] * ctx.block, M)
        da = torch.empty_like(dout)
        _blocksparse_softmax_bwd[grid](da, da.stride(0), dout, dout.stride(0), out, out.stride(0), ctx.scale, lut, dr, ctx.rel_shape[-1], ctx.rel_strides[0], ctx.rel_strides[1], ctx.rel_strides[2], ctx.is_causal, BLOCK_SIZE=ctx.block, ROW_SIZE=next_power_of_2(ctx.maxlut), IS_DENSE=ctx.is_dense, num_warps=num_warps(ctx.maxlut))
        return (da, None, None, dr, None, None, None, None, None, None, None, None, None, None, None, None, None, None)