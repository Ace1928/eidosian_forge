from boto.exception import BotoServerError
class TooManyRequestsException(BotoServerError):
    pass