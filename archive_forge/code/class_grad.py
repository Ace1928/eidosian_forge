import warnings
from autograd import jacobian as _jacobian
from autograd.core import make_vjp as _make_vjp
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.extend import vspace
from autograd.wrap_util import unary_to_nary
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError
class grad:
    """Returns the gradient as a callable function of hybrid quantum-classical functions.
    :func:`~.qjit` and Autograd compatible.

    By default, gradients are computed for arguments which contain
    the property ``requires_grad=True``. Alternatively, the ``argnum`` keyword argument
    can be specified to compute gradients for function arguments without this property,
    such as scalars, lists, tuples, dicts, or vanilla NumPy arrays. Setting
    ``argnum`` to the index of an argument with ``requires_grad=False`` will raise
    a ``NonDifferentiableError``.

    When the output gradient function is executed, both the forward pass
    *and* the backward pass will be performed in order to compute the gradient.
    The value of the forward pass is available via the :attr:`~.forward` property.

    .. warning::
        ``grad`` is intended to be used with the Autograd interface only.

    .. note::

        When used with :func:`~.qjit`, this function currently only supports the
        Catalyst compiler. See :func:`catalyst.grad` for more details.

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of the differences between Catalyst and PennyLane.

    Args:
        func (function): a plain QNode, or a Python function that contains
            a combination of quantum and classical nodes

        argnum (int, list(int), None): Which argument(s) to take the gradient
            with respect to. By default, the arguments themselves are used
            to determine differentiability, by examining the ``requires_grad``
            property.

        method (str): Specifies the gradient method when used with the :func:`~.qjit`
            decorator. Outside of :func:`~.qjit`, this keyword argument
            has no effect and should not be set. In just-in-time (JIT) mode,
            this can be any of ``["auto", "fd"]``, where:

            - ``"auto"`` represents deferring the quantum differentiation to the method
              specified by the QNode, while the classical computation is differentiated
              using traditional auto-diff. Catalyst supports ``"parameter-shift"`` and
              ``"adjoint"`` on internal QNodes. QNodes with ``diff_method="finite-diff"``
              are not supported with ``"auto"``.

            - ``"fd"`` represents first-order finite-differences for the entire hybrid
              function.

        h (float): The step-size value for the finite-difference (``"fd"``) method within
            :func:`~.qjit` decorated functions. This value has
            no effect in non-compiled functions.

    Returns:
        function: The function that returns the gradient of the input
        function with respect to the differentiable arguments, or, if specified,
        the arguments in ``argnum``.
    """

    def __new__(cls, func, argnum=None, method=None, h=None):
        """Patch to the proper grad function"""
        if (active_jit := compiler.active_compiler()):
            available_eps = compiler.AvailableCompilers.names_entrypoints
            ops_loader = available_eps[active_jit]['ops'].load()
            return ops_loader.grad(func, method=method, h=h, argnum=argnum)
        if method or h:
            raise ValueError(f"Invalid values for 'method={method}' and 'h={h}' in interpreted mode")
        return super().__new__(cls)

    def __init__(self, func, argnum=None):
        self._forward = None
        self._grad_fn = None
        self._fun = func
        self._argnum = argnum
        if self._argnum is not None:
            self._grad_fn = self._grad_with_forward(func, argnum=argnum)

    def _get_grad_fn(self, args):
        """Get the required gradient function.

        * If the differentiable argnum was provided on initialization,
          this has been pre-computed and is available via self._grad_fn

        * Otherwise, we must dynamically construct the gradient function by
          inspecting as to which of the parameter arguments are marked
          as differentiable.
        """
        if self._grad_fn is not None:
            return (self._grad_fn, self._argnum)
        argnum = []
        for idx, arg in enumerate(args):
            trainable = getattr(arg, 'requires_grad', None) or isinstance(arg, ArrayBox)
            if trainable:
                if arg.dtype.name[:3] == 'int':
                    raise ValueError('Autograd does not support differentiation of ints.')
                argnum.append(idx)
        if len(argnum) == 1:
            argnum = argnum[0]
        return (self._grad_with_forward(self._fun, argnum=argnum), argnum)

    def __call__(self, *args, **kwargs):
        """Evaluates the gradient function, and saves the function value
        calculated during the forward pass in :attr:`.forward`."""
        grad_fn, argnum = self._get_grad_fn(args)
        if not isinstance(argnum, int) and (not argnum):
            warnings.warn("Attempted to differentiate a function with no trainable parameters. If this is unintended, please add trainable parameters via the 'requires_grad' attribute or 'argnum' keyword.")
            self._forward = self._fun(*args, **kwargs)
            return ()
        grad_value, ans = grad_fn(*args, **kwargs)
        self._forward = ans
        return grad_value

    @property
    def forward(self):
        """float: The result of the forward pass calculated while performing
        backpropagation. Will return ``None`` if the backpropagation has not yet
        been performed."""
        return self._forward

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