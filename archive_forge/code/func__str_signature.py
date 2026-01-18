import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _str_signature(self):
    if self['Signature']:
        return [self['Signature'].replace('*', '\\*')] + ['']
    else:
        return ['']