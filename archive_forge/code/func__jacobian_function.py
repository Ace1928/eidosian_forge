import warnings
from autograd import jacobian as _jacobian
from autograd.core import make_vjp as _make_vjp
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.extend import vspace
from autograd.wrap_util import unary_to_nary
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError
def _jacobian_function(*args, **kwargs):
    """Compute the autograd Jacobian.

        This wrapper function is returned to the user instead of autograd.jacobian,
        so that we can take into account cases where the user computes the
        jacobian function once, but then calls it with arguments that change
        in differentiability.
        """
    if argnum is None:
        _argnum = _get_argnum(args)
        unpack = len(_argnum) == 1
    else:
        unpack = isinstance(argnum, int)
        _argnum = [argnum] if unpack else argnum
    if not _argnum:
        warnings.warn("Attempted to differentiate a function with no trainable parameters. If this is unintended, please add trainable parameters via the 'requires_grad' attribute or 'argnum' keyword.")
    jac = tuple((_jacobian(func, arg)(*args, **kwargs) for arg in _argnum))
    return jac[0] if unpack else jac