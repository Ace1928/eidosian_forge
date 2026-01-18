from the command line:
import functools
import re
import types
import unittest
import uuid
@functools.wraps(test_method)
def BoundParamTest(self):
    if isinstance(testcase_params, collections_abc.Mapping):
        test_method(self, **testcase_params)
    elif _NonStringIterable(testcase_params):
        test_method(self, *testcase_params)
    else:
        test_method(self, testcase_params)