from the command line:
import functools
import re
import types
import unittest
import uuid
def _UpdateClassDictForParamTestCase(dct, id_suffix, name, iterator):
    """Adds individual test cases to a dictionary.

  Args:
    dct: The target dictionary.
    id_suffix: The dictionary for mapping names to test IDs.
    name: The original name of the test case.
    iterator: The iterator generating the individual test cases.
  """
    for idx, func in enumerate(iterator):
        assert callable(func), 'Test generators must yield callables, got %r' % (func,)
        if getattr(func, '__x_use_name__', False):
            new_name = func.__name__
        else:
            new_name = '%s%s%d' % (name, _SEPARATOR, idx)
        assert new_name not in dct, 'Name of parameterized test case "%s" not unique' % (new_name,)
        dct[new_name] = func
        id_suffix[new_name] = getattr(func, '__x_extra_id__', '')