from boto.exception import BotoServerError
class InvalidPaymentInstrument(ResponseError):
    """The payment method used in the transaction is invalid.
    """