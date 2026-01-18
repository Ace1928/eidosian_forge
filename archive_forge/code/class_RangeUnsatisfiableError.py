class RangeUnsatisfiableError(ValueError):
    """Exception class when a byte range lies outside the file size boundaries."""

    def __init__(self, reason=None):
        if not reason:
            reason = 'Range is unsatisfiable'
        ValueError.__init__(self, reason)