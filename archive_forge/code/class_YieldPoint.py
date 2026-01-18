from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
class YieldPoint(object):

    def __init__(self, block, inst):
        assert isinstance(block, ir.Block)
        assert isinstance(inst, ir.Yield)
        self.block = block
        self.inst = inst
        self.live_vars = None
        self.weak_live_vars = None