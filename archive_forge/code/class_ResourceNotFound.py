from zaqarclient import errors
class ResourceNotFound(TransportError):
    """Indicates that a resource is missing

    This error maps to HTTP's 404
    """
    code = 404