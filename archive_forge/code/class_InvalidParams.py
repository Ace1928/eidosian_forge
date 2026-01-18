from boto.exception import BotoServerError
class InvalidParams(ResponseError):
    """One or more parameters in the request is invalid.
    """