from boto.exception import BotoServerError
class TransactionFullyRefundedAlready(ResponseError):
    """The transaction has already been completely refunded.
    """