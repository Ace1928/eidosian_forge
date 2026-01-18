import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def MergeLists(to, fro, to_file, fro_file, is_paths=False, append=True):

    def is_hashable(val):
        return val.__hash__

    def is_in_set_or_list(x, s, items):
        if is_hashable(x):
            return x in s
        return x in items
    prepend_index = 0
    hashable_to_set = {x for x in to if is_hashable(x)}
    for item in fro:
        singleton = False
        if type(item) in (str, int):
            if is_paths:
                to_item = MakePathRelative(to_file, fro_file, item)
            else:
                to_item = item
            if not (type(item) is str and item.startswith('-')):
                singleton = True
        elif type(item) is dict:
            to_item = {}
            MergeDicts(to_item, item, to_file, fro_file)
        elif type(item) is list:
            to_item = []
            MergeLists(to_item, item, to_file, fro_file)
        else:
            raise TypeError('Attempt to merge list item of unsupported type ' + item.__class__.__name__)
        if append:
            if not singleton or not is_in_set_or_list(to_item, hashable_to_set, to):
                to.append(to_item)
                if is_hashable(to_item):
                    hashable_to_set.add(to_item)
        else:
            while singleton and to_item in to:
                to.remove(to_item)
            to.insert(prepend_index, to_item)
            if is_hashable(to_item):
                hashable_to_set.add(to_item)
            prepend_index = prepend_index + 1