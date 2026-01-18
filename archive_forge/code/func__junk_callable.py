import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def _junk_callable():
    """A simple routine to profile."""
    result = sorted(['abc', 'def', 'ghi'])