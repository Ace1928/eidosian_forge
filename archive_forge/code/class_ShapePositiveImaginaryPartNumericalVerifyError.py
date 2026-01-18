class ShapePositiveImaginaryPartNumericalVerifyError(InequalityNumericalVerifyError, ShapeType):
    """
    Failed numerical verification of a shape having positive imaginary part.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'Numerical verification that shape has positive imaginary part has failed: Im(%r) > 0' % self.value