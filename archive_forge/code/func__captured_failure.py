import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def _captured_failure(msg):
    try:
        raise RuntimeError(msg)
    except Exception:
        return failure.Failure()