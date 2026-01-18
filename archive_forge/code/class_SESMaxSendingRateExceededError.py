from boto.exception import BotoServerError
class SESMaxSendingRateExceededError(SESError):
    """
    Your account's requests/second limit has been exceeded.
    """
    pass