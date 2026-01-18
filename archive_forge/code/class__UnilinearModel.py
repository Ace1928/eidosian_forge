import numpy as np
from scipy.odr._odrpack import Model
class _UnilinearModel(Model):
    """
    Univariate linear model

    This model is defined by :math:`y = \\beta_0 x + \\beta_1`

    Examples
    --------
    We can calculate orthogonal distance regression with an unilinear model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = 1.0 * x + 2.0
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.unilinear)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [1. 2.]

    """

    def __init__(self):
        super().__init__(_unilin, fjacd=_unilin_fjd, fjacb=_unilin_fjb, estimate=_unilin_est, meta={'name': 'Univariate Linear', 'equ': 'y = B_0 * x + B_1', 'TeXequ': '$y = \\beta_0 x + \\beta_1$'})