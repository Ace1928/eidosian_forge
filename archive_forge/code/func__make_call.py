from testtools.matchers import *
from ...tests import CapturedCall, TestCase
from ..smart.client import CallHookParams
from .matchers import *
def _make_call(self, method, args):
    return CapturedCall(CallHookParams(method, args, None, None, None), 0)