class InaccessibleSourceCodeError(PyCTError, ValueError):
    """Raised when inspect can not access source code."""