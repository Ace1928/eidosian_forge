from boto.exception import BotoServerError
class InvalidAccountState_Sender(RetriableResponseError):
    """Sender account cannot participate in the transaction.
    """