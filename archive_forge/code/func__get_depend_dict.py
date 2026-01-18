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
def _get_depend_dict(name, vars, deps):
    if name in vars:
        words = vars[name].get('depend', [])
        if '=' in vars[name] and (not isstring(vars[name])):
            for word in word_pattern.findall(vars[name]['=']):
                if word not in words and word in vars and (word != name):
                    words.append(word)
        for word in words[:]:
            for w in deps.get(word, []) or _get_depend_dict(word, vars, deps):
                if w not in words:
                    words.append(w)
    else:
        outmess('_get_depend_dict: no dependence info for %s\n' % repr(name))
        words = []
    deps[name] = words
    return words