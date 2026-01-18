from boto.exception import BotoServerError
class PaymentInstrumentNotCC(ResponseError):
    """The payment method specified in the transaction is not a credit
       card.  You can only use a credit card for this transaction.
    """