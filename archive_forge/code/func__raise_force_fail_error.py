import sys
from testtools.testresult import ExtendedToOriginalDecorator
def _raise_force_fail_error():
    raise AssertionError('Forced Test Failure')