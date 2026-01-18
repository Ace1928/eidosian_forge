from boto.exception import BotoServerError
class BaseException(BotoServerError):
    """
    A generic server-side error.
    """
    pass