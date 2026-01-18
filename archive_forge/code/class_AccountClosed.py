from boto.exception import BotoServerError
class AccountClosed(RetriableResponseError):
    """Account is not active.
    """