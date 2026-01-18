from boto.exception import BotoServerError
class InvalidAccountState_Caller(RetriableResponseError):
    """The developer account cannot participate in the transaction.
    """