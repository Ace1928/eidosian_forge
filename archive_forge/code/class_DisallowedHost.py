import operator
from django.utils.hashable import make_hashable
class DisallowedHost(SuspiciousOperation):
    """HTTP_HOST header contains invalid value"""
    pass