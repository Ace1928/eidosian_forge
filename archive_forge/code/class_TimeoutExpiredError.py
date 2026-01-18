from ncclient import NCClientError
class TimeoutExpiredError(NCClientError):
    pass