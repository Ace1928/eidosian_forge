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
def _build_marker(self, pos):
    source_desc, line, col = pos
    assert isinstance(source_desc, SourceDescriptor)
    contents = self.globalstate.commented_file_contents(source_desc)
    lines = contents[max(0, line - 3):line]
    lines[-1] += u'             # <<<<<<<<<<<<<<'
    lines += contents[line:line + 2]
    return u'"%s":%d\n%s\n' % (source_desc.get_escaped_description(), line, u'\n'.join(lines))