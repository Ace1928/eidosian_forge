from django.core.exceptions import SuspiciousOperation
class AlreadyRegistered(Exception):
    """The model is already registered."""
    pass