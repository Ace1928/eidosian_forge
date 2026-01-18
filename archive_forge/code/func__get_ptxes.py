from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def _get_ptxes(self, cc=None):
    if not cc:
        ctx = devices.get_context()
        device = ctx.device
        cc = device.compute_capability
    ptxes = self._ptx_cache.get(cc, None)
    if ptxes:
        return ptxes
    arch = nvvm.get_arch_option(*cc)
    options = self._nvvm_options.copy()
    options['arch'] = arch
    irs = self.llvm_strs
    ptxes = [nvvm.llvm_to_ptx(irs, **options)]
    ptxes = [x.decode().strip('\x00').strip() for x in ptxes]
    if config.DUMP_ASSEMBLY:
        print(('ASSEMBLY %s' % self._name).center(80, '-'))
        print(self._join_ptxes(ptxes))
        print('=' * 80)
    self._ptx_cache[cc] = ptxes
    return ptxes