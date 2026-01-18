import warnings
from autograd import jacobian as _jacobian
from autograd.core import make_vjp as _make_vjp
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.extend import vspace
from autograd.wrap_util import unary_to_nary
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError
def _get_argnum(args):
    """Inspect the arguments for differentiability and return the
        corresponding indices."""
    argnum = []
    for idx, arg in enumerate(args):
        trainable = getattr(arg, 'requires_grad', None) or isinstance(arg, ArrayBox)
        if trainable:
            if arg.dtype.name[:3] == 'int':
                raise ValueError('Autograd does not support differentiation of ints.')
            argnum.append(idx)
    return argnum