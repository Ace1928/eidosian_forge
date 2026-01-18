from boto.exception import BotoServerError
class InvalidTransactionId(ResponseError):
    """The specified transaction could not be found or the caller did not
       execute the transaction or this is not a Pay or Reserve call.
    """