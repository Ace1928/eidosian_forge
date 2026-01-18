from boto.exception import BotoServerError
class DisabledOperationException(BotoServerError):
    """
    Raised when an operation has been disabled.
    """
    pass