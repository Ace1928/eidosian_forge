from the command line:
import functools
import re
import types
import unittest
import uuid
def MakeBoundParamTest(testcase_params):

    @functools.wraps(test_method)
    def BoundParamTest(self):
        if isinstance(testcase_params, collections_abc.Mapping):
            test_method(self, **testcase_params)
        elif _NonStringIterable(testcase_params):
            test_method(self, *testcase_params)
        else:
            test_method(self, testcase_params)
    if naming_type is _FIRST_ARG:
        BoundParamTest.__x_use_name__ = True
        BoundParamTest.__name__ += str(testcase_params[0])
        testcase_params = testcase_params[1:]
    elif naming_type is _ARGUMENT_REPR:
        BoundParamTest.__x_extra_id__ = '(%s)' % (_FormatParameterList(testcase_params),)
    else:
        raise RuntimeError('%s is not a valid naming type.' % (naming_type,))
    BoundParamTest.__doc__ = '%s(%s)' % (BoundParamTest.__name__, _FormatParameterList(testcase_params))
    if test_method.__doc__:
        BoundParamTest.__doc__ += '\n%s' % (test_method.__doc__,)
    return BoundParamTest