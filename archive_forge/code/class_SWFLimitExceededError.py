from boto.exception import SWFResponseError
class SWFLimitExceededError(SWFResponseError):
    """
    Raised when when a system imposed limitation has been reached.
    """
    pass