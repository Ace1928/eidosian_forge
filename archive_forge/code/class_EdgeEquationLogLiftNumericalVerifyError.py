class EdgeEquationLogLiftNumericalVerifyError(LogLiftNumericalVerifyError, EdgeEquationType):
    """
    Exception for failed numerical verification that a logarithmic edge
    equation has error bound by epsilon.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'Numerical verification that logarthmic edge equation has small error failed: %r == 2 Pi I' % self.value