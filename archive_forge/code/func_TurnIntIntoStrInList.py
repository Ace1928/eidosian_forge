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
def TurnIntIntoStrInList(the_list):
    """Given list the_list, recursively converts all integers into strings.
  """
    for index, item in enumerate(the_list):
        if type(item) is int:
            the_list[index] = str(item)
        elif type(item) is dict:
            TurnIntIntoStrInDict(item)
        elif type(item) is list:
            TurnIntIntoStrInList(item)