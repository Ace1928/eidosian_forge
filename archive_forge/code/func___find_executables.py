import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def __find_executables(path):
    """Used by find_graphviz

    path - single directory as a string

    If any of the executables are found, it will return a dictionary
    containing the program names as keys and their paths as values.

    Otherwise returns None
    """
    success = False
    progs = {'dot': '', 'twopi': '', 'neato': '', 'circo': '', 'fdp': ''}
    was_quoted = False
    path = path.strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
        was_quoted = True
    if os.path.isdir(path):
        for prg in progs:
            if progs[prg]:
                continue
            if os.path.exists(os.path.join(path, prg)):
                if was_quoted:
                    progs[prg] = '"' + os.path.join(path, prg) + '"'
                else:
                    progs[prg] = os.path.join(path, prg)
                success = True
            elif os.path.exists(os.path.join(path, prg + '.exe')):
                if was_quoted:
                    progs[prg] = '"' + os.path.join(path, prg + '.exe') + '"'
                else:
                    progs[prg] = os.path.join(path, prg + '.exe')
                success = True
    if success:
        return progs
    else:
        return None