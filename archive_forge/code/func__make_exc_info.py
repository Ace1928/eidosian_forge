import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def _make_exc_info(msg):
    try:
        raise RuntimeError(msg)
    except Exception:
        return sys.exc_info()