from the command line:
import functools
import re
import types
import unittest
import uuid
def _FormatParameterList(testcase_params):
    if isinstance(testcase_params, collections_abc.Mapping):
        return ', '.join(('%s=%s' % (argname, _CleanRepr(value)) for argname, value in testcase_params.items()))
    elif _NonStringIterable(testcase_params):
        return ', '.join(map(_CleanRepr, testcase_params))
    else:
        return _FormatParameterList((testcase_params,))