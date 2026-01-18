from the base class `SymbolDoc`, and put the extra doc as the docstring
import re as _re
from .base import build_param_doc as _build_param_doc
class DropoutDoc(SymbolDoc):
    """
    Examples
    --------
    Apply dropout to corrupt input as zero with probability 0.2:

    >>> data = Variable('data')
    >>> data_dp = Dropout(data=data, p=0.2)

    >>> shape = (100, 100)  # take larger shapes to be more statistical stable
    >>> x = np.ones(shape)
    >>> op = Dropout(p=0.5, name='dp')
    >>> # dropout is identity during testing
    >>> y = test_utils.simple_forward(op, dp_data=x, is_train=False)
    >>> test_utils.almost_equal(x, y)
    True
    >>> y = test_utils.simple_forward(op, dp_data=x, is_train=True)
    >>> # expectation is (approximately) unchanged
    >>> np.abs(x.mean() - y.mean()) < 0.1
    True
    >>> set(np.unique(y)) == set([0, 2])
    True
    """