import sys
class NumpyRequiredForThisFeature(RuntimeError):
    """
    Error raised when user tries to use a feature that
    requires numpy without having numpy installed.
    """
    pass