class TiltProvenPositiveNumericalVerifyError(TiltInequalityNumericalVerifyError):
    """
    Numerically verifying that a tilt is negative has not only failed, we
    proved that the tilt is positive and thus that this cannot be a
    proto-canonical triangulation.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'Numerical verification that tilt is negative has failed, tilt is actually positive. This is provably not the proto-canonical triangulation: %r <= 0' % self.value