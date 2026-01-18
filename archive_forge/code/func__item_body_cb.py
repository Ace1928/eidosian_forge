import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
def _item_body_cb(self, indent, obj):
    _doc = obj._doc or obj._description or ''
    if not _doc:
        return ''
    wraplines = '\n ' not in _doc
    _doc = _item_body_formatter(_doc).rstrip()
    if not _doc:
        return ''
    _indent = indent + ' ' * self.indent_spacing
    if wraplines:
        doc_lines = textwrap.wrap(_doc, self.width, initial_indent=_indent, subsequent_indent=_indent)
        self.out.write('\n'.join(doc_lines).rstrip() + '\n')
    elif _doc.lstrip() == _doc:
        self.out.write(_indent + _doc + '\n')
    else:
        self.out.write(_doc + '\n')