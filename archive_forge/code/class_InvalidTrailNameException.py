from boto.exception import BotoServerError
class InvalidTrailNameException(BotoServerError):
    """
    Raised when the trail name is invalid.
    """
    pass