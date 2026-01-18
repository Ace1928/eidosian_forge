from boto.exception import BotoServerError
class MaximumNumberOfTrailsExceededException(BotoServerError):
    """
    Raised when no more trails can be created.
    """
    pass