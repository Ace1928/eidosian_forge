import functools
import os
import time
import uuid
from testtools import testcase
def _wip(f):

    @functools.wraps(f)
    def run_test(*args, **kwargs):
        __e = None
        try:
            f(*args, **kwargs)
        except Exception as __e:
            if expected_exception != Exception and (not isinstance(__e, expected_exception)):
                raise AssertionError('Work In Progress Test Failed%(bugstr)s with unexpected exception. Expected "%(expected)s" got "%(exception)s": %(message)s ' % {'message': message, 'bugstr': bugstr, 'expected': expected_exception.__class__.__name__, 'exception': __e.__class__.__name__})
            raise testcase.TestSkipped('Work In Progress Test Failed as expected%(bugstr)s: %(message)s' % {'message': message, 'bugstr': bugstr})
        raise AssertionError('Work In Progress Test Passed%(bugstr)s: %(message)s' % {'message': message, 'bugstr': bugstr})
    return run_test