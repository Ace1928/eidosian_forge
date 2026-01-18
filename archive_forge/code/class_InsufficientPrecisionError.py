class InsufficientPrecisionError(Exception):
    """
    This exception is raised when a computation fails and is likely
    to succeed if higher precision is used.
    """