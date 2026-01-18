import operator
from django.utils.hashable import make_hashable
class SynchronousOnlyOperation(Exception):
    """The user tried to call a sync-only function from an async context."""
    pass