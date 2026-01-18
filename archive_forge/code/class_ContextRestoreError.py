import threading
from itertools import count
class ContextRestoreError(Exception):
    """Raised when something is restored out-of-order."""