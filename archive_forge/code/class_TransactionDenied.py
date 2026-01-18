from boto.exception import BotoServerError
class TransactionDenied(ResponseError):
    """The transaction is not allowed.
    """