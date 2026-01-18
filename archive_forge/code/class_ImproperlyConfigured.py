import operator
from django.utils.hashable import make_hashable
class ImproperlyConfigured(Exception):
    """Django is somehow improperly configured"""
    pass