import torch
from .. import heuristics, jit
from .. import language as tl
from .. import next_power_of_2
@heuristics({'num_warps': lambda nargs: num_warps(nargs['N'])})
@heuristics({'BLOCK': lambda nargs: next_power_of_2(nargs['N'])})
@jit
def _backward(PROBS, IDX, DPROBS, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    idx = tl.load(IDX + row)
    PROBS = PROBS + row * N + cols
    probs = -tl.load(PROBS, mask=cols < N, other=float('inf'))
    probs = tl.exp(probs.to(tl.float32))
    delta = cols == idx
    dout = tl.load(DPROBS + row)
    din = (probs - delta) * dout
    tl.store(PROBS, din.to(PROBS.dtype.element_ty), mask=cols < N)