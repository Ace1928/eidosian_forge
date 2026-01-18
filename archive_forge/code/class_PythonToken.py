from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, \
from parso.python.token import PythonTokenTypes
from parso.utils import split_lines, PythonVersionInfo, parse_version_string
class PythonToken(Token):

    def __repr__(self):
        return 'TokenInfo(type=%s, string=%r, start_pos=%r, prefix=%r)' % self._replace(type=self.type.name)