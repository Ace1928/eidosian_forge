class ConsistencyWithSnapPeaNumericalVerifyError(NumericalVerifyError, ConsistencyWithSnapPeaType):
    """
    Exception raised when there is a significant numerical difference
    between the values computed by the SnapPea kernel and by this module
    for a given quantity.
    """

    def __init__(self, value, snappea_value):
        self.value = value
        self.snappea_value = snappea_value

    def __str__(self):
        return 'Inconsistency between SnapPea kernel and verify: %r == %r' % (self.snappea_value, self.value)