from the base class `SymbolDoc`, and put the extra doc as the docstring
import re as _re
from .base import build_param_doc as _build_param_doc
class FlattenDoc(SymbolDoc):
    """
    Examples
    --------
    Flatten is usually applied before `FullyConnected`, to reshape the 4D tensor
    produced by convolutional layers to 2D matrix:

    >>> data = Variable('data')  # say this is 4D from some conv/pool
    >>> flatten = Flatten(data=data, name='flat')  # now this is 2D
    >>> SymbolDoc.get_output_shape(flatten, data=(2, 3, 4, 5))
    {'flat_output': (2L, 60L)}

    >>> test_dims = [(2, 3, 4, 5), (2, 3), (2,)]
    >>> op = Flatten(name='flat')
    >>> for dims in test_dims:
    ...     x = test_utils.random_arrays(dims)
    ...     y = test_utils.simple_forward(op, flat_data=x)
    ...     y_np = x.reshape((dims[0], np.prod(dims[1:]).astype('int32')))
    ...     print('%s: %s' % (dims, test_utils.almost_equal(y, y_np)))
    (2, 3, 4, 5): True
    (2, 3): True
    (2,): True
    """