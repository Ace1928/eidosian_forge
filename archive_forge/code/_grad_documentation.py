import warnings
from autograd import jacobian as _jacobian
from autograd.core import make_vjp as _make_vjp
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.extend import vspace
from autograd.wrap_util import unary_to_nary
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError
Compute the autograd Jacobian.

        This wrapper function is returned to the user instead of autograd.jacobian,
        so that we can take into account cases where the user computes the
        jacobian function once, but then calls it with arguments that change
        in differentiability.
        