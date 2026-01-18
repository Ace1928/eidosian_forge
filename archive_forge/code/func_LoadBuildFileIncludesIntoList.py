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
def LoadBuildFileIncludesIntoList(sublist, sublist_path, data, aux_data, check):
    for item in sublist:
        if type(item) is dict:
            LoadBuildFileIncludesIntoDict(item, sublist_path, data, aux_data, None, check)
        elif type(item) is list:
            LoadBuildFileIncludesIntoList(item, sublist_path, data, aux_data, check)