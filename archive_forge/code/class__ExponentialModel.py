import numpy as np
from scipy.odr._odrpack import Model
class _ExponentialModel(Model):
    """
    Exponential model

    This model is defined by :math:`y=\\beta_0 + e^{\\beta_1 x}`

    Examples
    --------
    We can calculate orthogonal distance regression with an exponential model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = -10.0 + np.exp(0.5*x)
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.exponential)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [-10.    0.5]

    """

    def __init__(self):
        super().__init__(_exp_fcn, fjacd=_exp_fjd, fjacb=_exp_fjb, estimate=_exp_est, meta={'name': 'Exponential', 'equ': 'y= B_0 + exp(B_1 * x)', 'TeXequ': '$y=\\beta_0 + e^{\\beta_1 x}$'})