from django.core.exceptions import BadRequest, SuspiciousOperation
class SuspiciousSession(SuspiciousOperation):
    """The session may be tampered with"""
    pass