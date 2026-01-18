from zaqarclient import errors
class ServiceUnavailableError(TransportError):
    """Indicates that the server was unable to service the request

    This error maps to HTTP's 503
    """
    code = 503