from boto.exception import BotoServerError
class InvalidAccountState(RetriableResponseError):
    """The account is either suspended or closed.
    """