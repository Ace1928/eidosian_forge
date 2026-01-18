from ._internal import NDArrayBase
from ..base import _Null
def Activation(data=None, act_type=_Null, out=None, name=None, **kwargs):
    """Applies an activation function element-wise to the input.

    The following activation functions are supported:

    - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
    - `sigmoid`: :math:`y = \\frac{1}{1 + exp(-x)}`
    - `tanh`: Hyperbolic tangent, :math:`y = \\frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
    - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
    - `softsign`: :math:`y = \\frac{x}{1 + abs(x)}`



    Defined in ../src/operator/nn/activation.cc:L164

    Parameters
    ----------
    data : NDArray
        The input array.
    act_type : {'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'}, required
        Activation function to be applied.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)