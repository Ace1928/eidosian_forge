from contextlib import contextmanager
from copy import copy
class ContextPopException(Exception):
    """pop() has been called more times than push()"""
    pass