import torch
from ... import jit
from ... import language as tl
from ... import next_power_of_2
@jit
def _blocksparse_softmax_bwd(DA, stride_zdx, DOut, stride_zdout, Out, stride_zout, scale, LUT, DR, extent, stride_zr, stride_hr, stride_er, is_causal, ROW_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr, IS_DENSE: tl.constexpr):
    h = tl.program_id(0)
    m = tl.program_id(1)
    z = tl.program_id(2)
    hm = h * tl.num_programs(1) + m
    lane_n = tl.arange(0, ROW_SIZE) % BLOCK_SIZE
    block_n = tl.arange(0, ROW_SIZE) // BLOCK_SIZE
    header = LUT + hm // BLOCK_SIZE * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    off_mn = (offset + block_n) * BLOCK_SIZE * BLOCK_SIZE
    off_mn += m % BLOCK_SIZE * BLOCK_SIZE
    mask = block_n < size
    As = Out + z * stride_zout + off_mn
    DOuts = DOut + z * stride_zdout + off_mn
    if IS_DENSE:
        ns = tl.arange(0, ROW_SIZE)
    else:
        off_lut = offset + 2 * tl.num_programs(0) * tl.num_programs(1) // BLOCK_SIZE
        start_n = tl.load(LUT + off_lut + block_n, mask=mask, other=0)
        ns = start_n * BLOCK_SIZE + lane_n
    a = tl.load(As + lane_n, mask=mask, other=0.0)
    a = a.to(tl.float32)
    dout = tl.load(DOuts + lane_n, mask=mask, other=0.0)
    dout = dout.to(tl.float32)
    a = tl.where((ns > m) & is_causal & (a == a), 0.0, a)
    da = a * (dout - tl.sum(a * dout, 0))
    if DR is not None:
        DR += z * stride_zr
        DR += h * stride_hr
        off_lo = extent - m - 1 + ns
        mask_lo = (off_lo >= 0) & (off_lo < extent) & mask
        tl.store(DR + m * extent + off_lo, da, mask=mask_lo)
    da = da * scale
    DAs = DA + z * stride_zdx + off_mn
    tl.store(DAs + lane_n, da, mask=mask)