from django.core.exceptions import BadRequest, SuspiciousOperation
class SessionInterrupted(BadRequest):
    """The session was interrupted."""
    pass