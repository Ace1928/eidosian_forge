from boto.exception import BotoServerError
class SenderNotOriginalRecipient(ResponseError):
    """The sender in the refund transaction is not
       the recipient of the original transaction.
    """