import warnings
from autograd import jacobian as _jacobian
from autograd.core import make_vjp as _make_vjp
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.extend import vspace
from autograd.wrap_util import unary_to_nary
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError
@staticmethod
@unary_to_nary
def _grad_with_forward(fun, x):
    """This function is a replica of ``autograd.grad``, with the only
        difference being that it returns both the gradient *and* the forward pass
        value."""
    vjp, ans = _make_vjp(fun, x)
    if vspace(ans).size != 1:
        raise TypeError('Grad only applies to real scalar-output functions. Try jacobian, elementwise_grad or holomorphic_grad.')
    grad_value = vjp(vspace(ans).ones())
    return (grad_value, ans)