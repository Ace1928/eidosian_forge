from boto.exception import BotoServerError
class UnverifiedEmailAddress_Sender(ResponseError):
    """The sender account must have a verified
       email address for this payment.
    """