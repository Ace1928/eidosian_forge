from boto.exception import BotoServerError
class IdempotentParameterMismatchException(BotoServerError):
    pass