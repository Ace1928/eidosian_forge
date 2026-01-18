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
def MergeDicts(to, fro, to_file, fro_file):
    for k, v in fro.items():
        if k in to:
            bad_merge = False
            if type(v) in (str, int):
                if type(to[k]) not in (str, int):
                    bad_merge = True
            elif not isinstance(v, type(to[k])):
                bad_merge = True
            if bad_merge:
                raise TypeError('Attempt to merge dict value of type ' + v.__class__.__name__ + ' into incompatible type ' + to[k].__class__.__name__ + ' for key ' + k)
        if type(v) in (str, int):
            is_path = IsPathSection(k)
            if is_path:
                to[k] = MakePathRelative(to_file, fro_file, v)
            else:
                to[k] = v
        elif type(v) is dict:
            if k not in to:
                to[k] = {}
            MergeDicts(to[k], v, to_file, fro_file)
        elif type(v) is list:
            ext = k[-1]
            append = True
            if ext == '=':
                list_base = k[:-1]
                lists_incompatible = [list_base, list_base + '?']
                to[list_base] = []
            elif ext == '+':
                list_base = k[:-1]
                lists_incompatible = [list_base + '=', list_base + '?']
                append = False
            elif ext == '?':
                list_base = k[:-1]
                lists_incompatible = [list_base, list_base + '=', list_base + '+']
            else:
                list_base = k
                lists_incompatible = [list_base + '=', list_base + '?']
            for list_incompatible in lists_incompatible:
                if list_incompatible in fro:
                    raise GypError('Incompatible list policies ' + k + ' and ' + list_incompatible)
            if list_base in to:
                if ext == '?':
                    continue
                elif type(to[list_base]) is not list:
                    raise TypeError('Attempt to merge dict value of type ' + v.__class__.__name__ + ' into incompatible type ' + to[list_base].__class__.__name__ + ' for key ' + list_base + '(' + k + ')')
            else:
                to[list_base] = []
            is_paths = IsPathSection(list_base)
            MergeLists(to[list_base], v, to_file, fro_file, is_paths, append)
        else:
            raise TypeError('Attempt to merge dict value of unsupported type ' + v.__class__.__name__ + ' for key ' + k)