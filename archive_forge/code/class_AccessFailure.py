from boto.exception import BotoServerError
class AccessFailure(RetriableResponseError):
    """Account cannot be accessed.
    """