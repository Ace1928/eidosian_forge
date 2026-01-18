import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def _init_builtins(ctx):
    ctx.one = ctx.mpf(1)
    ctx.zero = ctx.mpf(0)
    ctx.inf = ctx.mpf('inf')
    ctx.ninf = -ctx.inf
    ctx.nan = ctx.mpf('nan')
    ctx.j = ctx.mpc(0, 1)
    ctx.exp = ctx._wrap_mpi_function(libmp.mpi_exp, libmp.mpci_exp)
    ctx.sqrt = ctx._wrap_mpi_function(libmp.mpi_sqrt)
    ctx.ln = ctx._wrap_mpi_function(libmp.mpi_log, libmp.mpci_log)
    ctx.cos = ctx._wrap_mpi_function(libmp.mpi_cos, libmp.mpci_cos)
    ctx.sin = ctx._wrap_mpi_function(libmp.mpi_sin, libmp.mpci_sin)
    ctx.tan = ctx._wrap_mpi_function(libmp.mpi_tan)
    ctx.gamma = ctx._wrap_mpi_function(libmp.mpi_gamma, libmp.mpci_gamma)
    ctx.loggamma = ctx._wrap_mpi_function(libmp.mpi_loggamma, libmp.mpci_loggamma)
    ctx.rgamma = ctx._wrap_mpi_function(libmp.mpi_rgamma, libmp.mpci_rgamma)
    ctx.factorial = ctx._wrap_mpi_function(libmp.mpi_factorial, libmp.mpci_factorial)
    ctx.fac = ctx.factorial
    ctx.eps = ctx._constant(lambda prec, rnd: (0, MPZ_ONE, 1 - prec, 1))
    ctx.pi = ctx._constant(libmp.mpf_pi)
    ctx.e = ctx._constant(libmp.mpf_e)
    ctx.ln2 = ctx._constant(libmp.mpf_ln2)
    ctx.ln10 = ctx._constant(libmp.mpf_ln10)
    ctx.phi = ctx._constant(libmp.mpf_phi)
    ctx.euler = ctx._constant(libmp.mpf_euler)
    ctx.catalan = ctx._constant(libmp.mpf_catalan)
    ctx.glaisher = ctx._constant(libmp.mpf_glaisher)
    ctx.khinchin = ctx._constant(libmp.mpf_khinchin)
    ctx.twinprime = ctx._constant(libmp.mpf_twinprime)