import operator
from django.utils.hashable import make_hashable
class SuspiciousFileOperation(SuspiciousOperation):
    """A Suspicious filesystem operation was attempted"""
    pass