import hashlib
class SignatureError(CommError):
    """Unknown user, signature on msg invalid, or not within allowed time
    range."""
    pass