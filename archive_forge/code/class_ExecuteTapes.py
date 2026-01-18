import inspect
import logging
import numpy as np
import torch
import torch.utils._pytree as pytree
import pennylane as qml
@pytreeify
class ExecuteTapes(torch.autograd.Function):
    """The signature of this ``torch.autograd.Function`` is designed to
    work around Torch restrictions.

    In particular, ``torch.autograd.Function``:

    - Cannot accept keyword arguments. As a result, we pass a dictionary
      as the first argument ``kwargs``. This dictionary **must** contain:

      * ``"tapes"``: the quantum tapes to batch evaluate
      * ``"execute_fn"``: a function that calculates the results of the tapes
      * ``"jpc"``: a :class:`~.JacobianProductCalculator` that can compute the vjp.

    Further, note that the ``parameters`` argument is dependent on the
    ``tapes``; this function should always be called
    with the parameters extracted directly from the tapes as follows:

    >>> parameters = [p for t in tapes for p in t.get_parameters()]
    >>> kwargs = {"tapes": tapes, "execute_fn": execute_fn, "jpc": jpc}
    >>> ExecuteTapes.apply(kwargs, *parameters)

    """

    @staticmethod
    def forward(ctx, kwargs, *parameters):
        """Implements the forward pass batch tape evaluation."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Entry with args=(ctx=%s, kwargs=%s, parameters=%s) called by=%s', ctx, kwargs, parameters, '::L'.join((str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3])))
        ctx.tapes = kwargs['tapes']
        ctx.jpc = kwargs['jpc']
        res = tuple(kwargs['execute_fn'](ctx.tapes))
        ctx.torch_device = None
        for p in parameters:
            if isinstance(p, torch.Tensor) and p.is_cuda:
                ctx.torch_device = p.get_device()
                break
        res = tuple((_res_to_torch(r, ctx) for r in res))
        return res

    @staticmethod
    def backward(ctx, *dy):
        """Returns the vector-Jacobian product with given
        parameter values p and output gradient dy"""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Entry with args=(ctx=%s, dy=%s) called by=%s', ctx, dy, '::L'.join((str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3])))
        dy = _recursive_conj(dy)
        vjps = ctx.jpc.compute_vjp(ctx.tapes, dy)
        unpacked_vjps = []
        for vjp_slice in vjps:
            if vjp_slice is not None and np.squeeze(vjp_slice).shape != (0,):
                unpacked_vjps.extend(_res_to_torch(vjp_slice, ctx))
        vjps = tuple(unpacked_vjps)
        return (None,) + vjps