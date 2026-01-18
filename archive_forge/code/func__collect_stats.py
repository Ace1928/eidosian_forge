import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def _collect_stats():
    """Collect and return some dummy profile data."""
    ret, stats = lsprof.profile(_junk_callable)
    return stats