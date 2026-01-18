from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
@classmethod
def _add_utility(cls, utility, type, lines, begin_lineno, tags=None):
    if utility is None:
        return
    code = '\n'.join(lines)
    if tags and 'substitute' in tags and ('naming' in tags['substitute']):
        try:
            code = Template(code).substitute(vars(Naming))
        except (KeyError, ValueError) as e:
            raise RuntimeError("Error parsing templated utility code of type '%s' at line %d: %s" % (type, begin_lineno, e))
    code = '\n' * begin_lineno + code
    if type == 'proto':
        utility[0] = code
    elif type == 'impl':
        utility[1] = code
    else:
        all_tags = utility[2]
        all_tags[type] = code
    if tags:
        all_tags = utility[2]
        for name, values in tags.items():
            all_tags.setdefault(name, set()).update(values)