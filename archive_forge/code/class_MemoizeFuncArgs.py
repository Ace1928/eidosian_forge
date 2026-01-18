from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
class MemoizeFuncArgs(dict):

    def __missing__(self, _key):
        self[_key] = func(*args, **kwargs)
        return self[_key]