import operator
from django.utils.hashable import make_hashable
class AppRegistryNotReady(Exception):
    """The django.apps registry is not populated yet"""
    pass