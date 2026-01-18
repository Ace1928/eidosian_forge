import hashlib
class IncompleteMessageError(ProtocolError):
    """A complete requested was not received."""
    pass