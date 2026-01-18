import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def _resolvetypedefpattern(line):
    line = ''.join(line.split())
    m1 = typedefpattern.match(line)
    print(line, m1)
    if m1:
        attrs = m1.group('attributes')
        attrs = [a.lower() for a in attrs.split(',')] if attrs else []
        return (m1.group('name'), attrs, m1.group('params'))
    return (None, [], None)