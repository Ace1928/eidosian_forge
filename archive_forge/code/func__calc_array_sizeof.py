from numba.cuda.cudadrv import devices, driver
from numba.core.registry import cpu_target
def _calc_array_sizeof(ndim):
    """
    Use the ABI size in the CPU target
    """
    ctx = cpu_target.target_context
    return ctx.calc_array_sizeof(ndim)