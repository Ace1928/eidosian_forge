import numpy as np
from scipy.odr._odrpack import Model
class _MultilinearModel(Model):
    """
    Arbitrary-dimensional linear model

    This model is defined by :math:`y=\\beta_0 + \\sum_{i=1}^m \\beta_i x_i`

    Examples
    --------
    We can calculate orthogonal distance regression with an arbitrary
    dimensional linear model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = 10.0 + 5.0 * x
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.multilinear)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [10.  5.]

    """

    def __init__(self):
        super().__init__(_lin_fcn, fjacb=_lin_fjb, fjacd=_lin_fjd, estimate=_lin_est, meta={'name': 'Arbitrary-dimensional Linear', 'equ': 'y = B_0 + Sum[i=1..m, B_i * x_i]', 'TeXequ': '$y=\\beta_0 + \\sum_{i=1}^m \\beta_i x_i$'})