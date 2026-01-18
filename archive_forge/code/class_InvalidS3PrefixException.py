from boto.exception import BotoServerError
class InvalidS3PrefixException(BotoServerError):
    """
    Raised when an invalid key prefix is given.
    """
    pass