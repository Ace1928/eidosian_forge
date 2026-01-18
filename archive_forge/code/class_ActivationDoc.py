from the base class `SymbolDoc`, and put the extra doc as the docstring
import re as _re
from .base import build_param_doc as _build_param_doc
class ActivationDoc(SymbolDoc):
    """
    Examples
    --------
    A one-hidden-layer MLP with ReLU activation:

    >>> data = Variable('data')
    >>> mlp = FullyConnected(data=data, num_hidden=128, name='proj')
    >>> mlp = Activation(data=mlp, act_type='relu', name='activation')
    >>> mlp = FullyConnected(data=mlp, num_hidden=10, name='mlp')
    >>> mlp
    <Symbol mlp>

    ReLU activation

    >>> test_suites = [
    ...     ('relu', lambda x: np.maximum(x, 0)),
    ...     ('sigmoid', lambda x: 1 / (1 + np.exp(-x))),
    ...     ('tanh', lambda x: np.tanh(x)),
    ...     ('softrelu', lambda x: np.log(1 + np.exp(x)))
    ... ]
    >>> x = test_utils.random_arrays((2, 3, 4))
    >>> for act_type, numpy_impl in test_suites:
    ...     op = Activation(act_type=act_type, name='act')
    ...     y = test_utils.simple_forward(op, act_data=x)
    ...     y_np = numpy_impl(x)
    ...     print('%s: %s' % (act_type, test_utils.almost_equal(y, y_np)))
    relu: True
    sigmoid: True
    tanh: True
    softrelu: True
    """