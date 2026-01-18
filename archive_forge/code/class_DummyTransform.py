from numpy.testing import assert_equal
import numpy as np
class DummyTransform:
    """Conversion between full rank dummy encodings


    y = X b + u
    b = C a
    a = C^{-1} b

    y = X C a + u

    define Z = X C, then

    y = Z a + u

    contrasts:

    R_b b = r

    R_a a = R_b C a = r

    where R_a = R_b C

    Here C is the transform matrix, with dot_left and dot_right as the main
    methods, and the same for the inverse transform matrix, C^{-1}

    Note:
     - The class was mainly written to keep left and right straight.
     - No checking is done.
     - not sure yet if method names make sense


    """

    def __init__(self, d1, d2):
        """C such that d1 C = d2, with d1 = X, d2 = Z

        should be (x, z) in arguments ?
        """
        self.transf_matrix = np.linalg.lstsq(d1, d2, rcond=-1)[0]
        self.invtransf_matrix = np.linalg.lstsq(d2, d1, rcond=-1)[0]

    def dot_left(self, a):
        """ b = C a
        """
        return np.dot(self.transf_matrix, a)

    def dot_right(self, x):
        """ z = x C
        """
        return np.dot(x, self.transf_matrix)

    def inv_dot_left(self, b):
        """ a = C^{-1} b
        """
        return np.dot(self.invtransf_matrix, b)

    def inv_dot_right(self, z):
        """ x = z C^{-1}
        """
        return np.dot(z, self.invtransf_matrix)