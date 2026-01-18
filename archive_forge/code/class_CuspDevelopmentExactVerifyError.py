class CuspDevelopmentExactVerifyError(ExactVerifyError, CuspDevelopmentType):
    """
    Raised when finding a consistent assignment of side lengths to the
    Euclidean Horotriangles to form a Euclidean Horotorus for a cusp failed
    using exact arithmetic.
    """

    def __init__(self, value1, value2):
        self.value1 = value1
        self.value2 = value2

    def __str__(self):
        return 'Inconsistency in the side lengths of the Euclidean Horotriangles for a cusp: %r = %r' % (self.value1, self.value2)