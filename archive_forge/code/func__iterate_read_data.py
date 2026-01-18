from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _iterate_read_data(read_data):
    sep = b'\n' if isinstance(read_data, bytes) else '\n'
    data_as_list = [l + sep for l in read_data.split(sep)]
    if data_as_list[-1] == sep:
        data_as_list = data_as_list[:-1]
    else:
        data_as_list[-1] = data_as_list[-1][:-1]
    for line in data_as_list:
        yield line