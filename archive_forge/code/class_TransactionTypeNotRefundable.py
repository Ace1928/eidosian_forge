from boto.exception import BotoServerError
class TransactionTypeNotRefundable(ResponseError):
    """You cannot refund this transaction.
    """