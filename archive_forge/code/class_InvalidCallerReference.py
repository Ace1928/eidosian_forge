from boto.exception import BotoServerError
class InvalidCallerReference(ResponseError):
    """The Caller Reference does not have a token associated with it.
    """