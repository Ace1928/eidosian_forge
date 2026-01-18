from zaqarclient import errors
class UnauthorizedError(TransportError):
    """Indicates that a request was not authenticated

    This error maps to HTTP's 401
    """
    code = 401