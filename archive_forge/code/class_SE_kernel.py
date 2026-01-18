import numpy as np
import numpy.linalg as la
class SE_kernel(Kernel):
    """Squared exponential kernel without derivatives"""

    def __init__(self):
        Kernel.__init__(self)

    def set_params(self, params):
        """Set the parameters of the squared exponential kernel.

        Parameters:

        params: [weight, l] Parameters of the kernel:
            weight: prefactor of the exponential
            l : scale of the kernel
        """
        self.weight = params[0]
        self.l = params[1]

    def squared_distance(self, x1, x2):
        """Returns the norm of x1-x2 using diag(l) as metric """
        return np.sum((x1 - x2) * (x1 - x2)) / self.l ** 2

    def kernel(self, x1, x2):
        """ This is the squared exponential function"""
        return self.weight ** 2 * np.exp(-0.5 * self.squared_distance(x1, x2))

    def dK_dweight(self, x1, x2):
        """Derivative of the kernel respect to the weight """
        return 2 * self.weight * np.exp(-0.5 * self.squared_distance(x1, x2))

    def dK_dl(self, x1, x2):
        """Derivative of the kernel respect to the scale"""
        return self.kernel * la.norm(x1 - x2) ** 2 / self.l ** 3