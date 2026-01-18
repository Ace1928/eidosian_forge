from boto.exception import BotoServerError
class TrailAlreadyExistsException(BotoServerError):
    """
    Raised when the given trail name already exists.
    """
    pass