from boto.exception import BotoServerError
class SESIdentityNotVerifiedError(SESError):
    """
    Raised when an identity (domain or address) has not been verified in SES yet.
    """
    pass