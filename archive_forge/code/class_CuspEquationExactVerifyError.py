class CuspEquationExactVerifyError(ExactVerifyError, CuspEquationType):
    """
    Exception for failed verification of a polynomial cusp gluing equation
    using exact arithmetics.
    """

    def __init__(self, value, expected_value):
        self.value = value

    def __str__(self):
        return 'Verification of a polynomial cusp equation using exact arithmetic failed: %r == 1' % self.value