import re
from django.utils.functional import SimpleLazyObject
class NonCapture(list):
    """Represent a non-capturing group in the pattern string."""