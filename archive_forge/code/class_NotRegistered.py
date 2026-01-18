from django.core.exceptions import SuspiciousOperation
class NotRegistered(Exception):
    """The model is not registered."""
    pass