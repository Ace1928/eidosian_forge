import torch
from ... import cdiv, heuristics, jit
from ... import language as tl
class _matmul(torch.autograd.Function):
    fn = {'sdd': sdd_matmul, 'dsd': dsd_matmul, 'dds': dds_matmul}

    @staticmethod
    def forward(ctx, a, b, trans_a, trans_b, trans_c, mode, spdims, block, c_lut, c_width, da_lut, da_width, db_lut, db_width, out):
        c = _matmul.fn[mode](a, b, trans_a, trans_b, trans_c, spdims, block, c_lut, c_width, out=out)
        ctx.save_for_backward(a, b)
        ctx.da_lut = da_lut
        ctx.da_width = da_width
        ctx.db_lut = db_lut
        ctx.db_width = db_width
        ctx.mode = mode
        ctx.spdims = spdims
        ctx.block = block
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.trans_c = trans_c
        ctx.has_out = out is not None
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da, db = (None, None)
        mode = ctx.mode
        if ctx.needs_input_grad[0]:
            mode_da = mode[1] + mode[0] + mode[2]
            da = _matmul.fn[mode_da](dc, b, ctx.trans_c, not ctx.trans_b, ctx.trans_a, ctx.spdims, ctx.block, ctx.da_lut, ctx.da_width)
        if ctx.needs_input_grad[1]:
            mode_db = mode[2] + mode[1] + mode[0]
            db = _matmul.fn[mode_db](a, dc, not ctx.trans_a, ctx.trans_c, ctx.trans_b, ctx.spdims, ctx.block, ctx.db_lut, ctx.db_width)
        dout = dc if ctx.has_out else None
        return (da, db, None, None, None, None, None, None, None, None, None, None, None, None, dout)