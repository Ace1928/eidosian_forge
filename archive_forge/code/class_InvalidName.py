import sys
from jsonschema.compat import PY3
class InvalidName(ValueError):
    """
    The given name is not a dot-separated list of Python objects.
    """