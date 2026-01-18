from ._internal import NDArrayBase
from ..base import _Null
def adam_update(weight=None, grad=None, mean=None, var=None, lr=_Null, beta1=_Null, beta2=_Null, epsilon=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, lazy_update=_Null, out=None, name=None, **kwargs):
    """Update function for Adam optimizer. Adam is seen as a generalization
    of AdaGrad.

    Adam update consists of the following steps, where g represents gradient and m, v
    are 1st and 2nd order moment estimates (mean and variance).

    .. math::

     g_t = \\nabla J(W_{t-1})\\\\
     m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t\\\\
     v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2\\\\
     W_t = W_{t-1} - \\alpha \\frac{ m_t }{ \\sqrt{ v_t } + \\epsilon }

    It updates the weights using::

     m = beta1*m + (1-beta1)*grad
     v = beta2*v + (1-beta2)*(grad**2)
     w += - learning_rate * m / (sqrt(v) + epsilon)

    However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and the storage
    type of weight is the same as those of m and v,
    only the row slices whose indices appear in grad.indices are updated (for w, m and v)::

     for row in grad.indices:
         m[row] = beta1*m[row] + (1-beta1)*grad[row]
         v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)
         w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)



    Defined in ../src/operator/optimizer_op.cc:L687

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    mean : NDArray
        Moving mean
    var : NDArray
        Moving variance
    lr : float, required
        Learning rate
    beta1 : float, optional, default=0.899999976
        The decay rate for the 1st moment estimates.
    beta2 : float, optional, default=0.999000013
        The decay rate for the 2nd moment estimates.
    epsilon : float, optional, default=9.99999994e-09
        A small constant for numerical stability.
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    lazy_update : boolean, optional, default=1
        If true, lazy updates are applied if gradient's stype is row_sparse and all of w, m and v have the same stype

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)