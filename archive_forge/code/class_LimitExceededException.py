from boto.exception import BotoServerError
class LimitExceededException(BotoServerError):
    pass