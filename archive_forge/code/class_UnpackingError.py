import pkgutil
import re
from jsbeautifier.unpackers import evalbased
class UnpackingError(Exception):
    """Badly packed source or general error. Argument is a
    meaningful description."""
    pass