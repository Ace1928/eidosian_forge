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
def FindEnclosingBracketGroup(input_str):
    stack = []
    start = -1
    for index, char in enumerate(input_str):
        if char in LBRACKETS:
            stack.append(char)
            if start == -1:
                start = index
        elif char in BRACKETS:
            if not stack:
                return (-1, -1)
            if stack.pop() != BRACKETS[char]:
                return (-1, -1)
            if not stack:
                return (start, index + 1)
    return (-1, -1)