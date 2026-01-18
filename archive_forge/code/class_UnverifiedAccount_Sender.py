from boto.exception import BotoServerError
class UnverifiedAccount_Sender(ResponseError):
    """The sender's account must have a verified U.S.  credit card or
       a verified U.S bank account before this transaction can be
       initiated.
    """