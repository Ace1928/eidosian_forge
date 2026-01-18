import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
def _add_tests_from_data(cls, name, func, data):
    """
    Add tests from data loaded from the data file into the class
    """
    index_len = len(str(len(data)))
    for i, elem in enumerate(data):
        if isinstance(data, dict):
            key, value = (elem, data[elem])
            test_name = mk_test_name(name, key, i, index_len)
        elif isinstance(data, list):
            value = elem
            test_name = mk_test_name(name, value, i, index_len)
        if isinstance(value, dict):
            add_test(cls, test_name, test_name, func, **value)
        else:
            add_test(cls, test_name, test_name, func, value)