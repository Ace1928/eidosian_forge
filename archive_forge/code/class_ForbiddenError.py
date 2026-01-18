from zaqarclient import errors
class ForbiddenError(TransportError):
    """Indicates that a request is forbidden to access the particular resource

    This error maps to HTTP's 403
    """
    code = 403