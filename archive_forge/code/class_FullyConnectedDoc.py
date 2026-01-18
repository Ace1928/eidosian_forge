from the base class `SymbolDoc`, and put the extra doc as the docstring
import re as _re
from .base import build_param_doc as _build_param_doc
class FullyConnectedDoc(SymbolDoc):
    """
    Examples
    --------
    Construct a fully connected operator with target dimension 512.

    >>> data = Variable('data')  # or some constructed NN
    >>> op = FullyConnected(data=data,
    ...                     num_hidden=512,
    ...                     name='FC1')
    >>> op
    <Symbol FC1>
    >>> SymbolDoc.get_output_shape(op, data=(128, 100))
    {'FC1_output': (128L, 512L)}

    A simple 3-layer MLP with ReLU activation:

    >>> net = Variable('data')
    >>> for i, dim in enumerate([128, 64]):
    ...     net = FullyConnected(data=net, num_hidden=dim, name='FC%d' % i)
    ...     net = Activation(data=net, act_type='relu', name='ReLU%d' % i)
    >>> # 10-class predictor (e.g. MNIST)
    >>> net = FullyConnected(data=net, num_hidden=10, name='pred')
    >>> net
    <Symbol pred>

    >>> dim_in, dim_out = (3, 4)
    >>> x, w, b = test_utils.random_arrays((10, dim_in), (dim_out, dim_in), (dim_out,))
    >>> op = FullyConnected(num_hidden=dim_out, name='FC')
    >>> out = test_utils.simple_forward(op, FC_data=x, FC_weight=w, FC_bias=b)
    >>> # numpy implementation of FullyConnected
    >>> out_np = np.dot(x, w.T) + b
    >>> test_utils.almost_equal(out, out_np)
    True
    """