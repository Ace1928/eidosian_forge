from ._internal import NDArrayBase
from ..base import _Null
def LRN(data=None, alpha=_Null, beta=_Null, knorm=_Null, nsize=_Null, out=None, name=None, **kwargs):
    """Applies local response normalization to the input.

    The local response normalization layer performs "lateral inhibition" by normalizing
    over local input regions.

    If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position
    :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
    activity :math:`b_{x,y}^{i}` is given by the expression:

    .. math::
       b_{x,y}^{i} = \\frac{a_{x,y}^{i}}{\\Bigg({k + \\frac{\\alpha}{n} \\sum_{j=max(0, i-\\frac{n}{2})}^{min(N-1, i+\\frac{n}{2})} (a_{x,y}^{j})^{2}}\\Bigg)^{\\beta}}

    where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the total
    number of kernels in the layer.



    Defined in ../src/operator/nn/lrn.cc:L157

    Parameters
    ----------
    data : NDArray
        Input data to LRN
    alpha : float, optional, default=9.99999975e-05
        The variance scaling parameter :math:`\x07lpha` in the LRN expression.
    beta : float, optional, default=0.75
        The power parameter :math:`\x08eta` in the LRN expression.
    knorm : float, optional, default=2
        The parameter :math:`k` in the LRN expression.
    nsize : int (non-negative), required
        normalization window width in elements.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)