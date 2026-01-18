import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
class SRegBuilder(object):

    def __init__(self, builder):
        self.builder = builder

    def tid(self, xyz):
        return call_sreg(self.builder, 'tid.%s' % xyz)

    def ctaid(self, xyz):
        return call_sreg(self.builder, 'ctaid.%s' % xyz)

    def ntid(self, xyz):
        return call_sreg(self.builder, 'ntid.%s' % xyz)

    def nctaid(self, xyz):
        return call_sreg(self.builder, 'nctaid.%s' % xyz)

    def getdim(self, xyz):
        i64 = ir.IntType(64)
        tid = self.builder.sext(self.tid(xyz), i64)
        ntid = self.builder.sext(self.ntid(xyz), i64)
        nctaid = self.builder.sext(self.ctaid(xyz), i64)
        res = self.builder.add(self.builder.mul(ntid, nctaid), tid)
        return res