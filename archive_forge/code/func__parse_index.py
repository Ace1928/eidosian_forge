import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _parse_index(self, section, content):
    """
        .. index:: default
           :refguide: something, else, and more

        """

    def strip_each_in(lst):
        return [s.strip() for s in lst]
    out = {}
    section = section.split('::')
    if len(section) > 1:
        out['default'] = strip_each_in(section[1].split(','))[0]
    for line in content:
        line = line.split(':')
        if len(line) > 2:
            out[line[1]] = strip_each_in(line[2].split(','))
    return out