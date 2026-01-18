from . import utilities
class ZeroDimensionalComponent(Component):
    """
    A list holding solutions of the Ptolemy variety belonging
    to the same primary zero-dimensional component of the
    variety (i.e., Galois conjugate solutions).
    """

    def __init__(self, l, p=None):
        self.dimension = 0
        super(ZeroDimensionalComponent, self).__init__(l)