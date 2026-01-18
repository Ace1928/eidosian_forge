from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def _numpy_not_installed_raise(*args, **kwargs):
    raise ImportError('numpy not installed, no such method')