import operator
from django.utils.hashable import make_hashable
class SuspiciousOperation(Exception):
    """The user did something suspicious"""