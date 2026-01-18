from ._internal import NDArrayBase
from ..base import _Null
def adagrad_update(weight=None, grad=None, history=None, lr=_Null, epsilon=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, out=None, name=None, **kwargs):
    """Update function for AdaGrad optimizer.

    Referenced from *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*,
    and available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.

    Updates are applied by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
        history = history + square(rescaled_grad)
        w = w - learning_rate * rescaled_grad / sqrt(history + epsilon)

    Note that non-zero values for the weight decay option are not supported.



    Defined in ../src/operator/optimizer_op.cc:L908

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    history : NDArray
        History
    lr : float, required
        Learning rate
    epsilon : float, optional, default=1.00000001e-07
        epsilon
    wd : float, optional, default=0
        weight decay
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)