from boto.exception import BotoServerError
class ExpiredIteratorException(BotoServerError):
    pass