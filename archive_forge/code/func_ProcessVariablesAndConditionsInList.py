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
def ProcessVariablesAndConditionsInList(the_list, phase, variables, build_file):
    index = 0
    while index < len(the_list):
        item = the_list[index]
        if type(item) is dict:
            ProcessVariablesAndConditionsInDict(item, phase, variables, build_file)
        elif type(item) is list:
            ProcessVariablesAndConditionsInList(item, phase, variables, build_file)
        elif type(item) is str:
            expanded = ExpandVariables(item, phase, variables, build_file)
            if type(expanded) in (str, int):
                the_list[index] = expanded
            elif type(expanded) is list:
                the_list[index:index + 1] = expanded
                index += len(expanded)
                continue
            else:
                raise ValueError('Variable expansion in this context permits strings and ' + 'lists only, found ' + expanded.__class__.__name__ + ' at ' + index)
        elif type(item) is not int:
            raise TypeError('Unknown type ' + item.__class__.__name__ + ' at index ' + index)
        index = index + 1