import sys as _sys
import pprint as _pprint
class NotRegisteredError(Exception):
    """
    An Exception indicating that an OUI or IAB was not found in the IEEE
    Registry.
    """
    pass