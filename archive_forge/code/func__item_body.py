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
def _item_body(self, indent, obj):
    typeinfo = ', '.join(filter(None, ['dict' if isinstance(obj, ConfigDict) else obj.domain_name(), 'optional' if obj._default is None else f'default={repr(obj._default)}']))
    self.out.write(f'\n{indent}{obj.name()}: {typeinfo}\n')
    self.wrapper.initial_indent = indent + ' ' * self.indent_spacing
    self.wrapper.subsequent_indent = indent + ' ' * self.indent_spacing
    vis = ''
    if self.visibility is None and obj._visibility >= ADVANCED_OPTION:
        vis = '[ADVANCED option]'
        if obj._visibility >= DEVELOPER_OPTION:
            vis = '[DEVELOPER option]'
    itemdoc = wrap_reStructuredText('\n\n'.join(filter(None, [vis, inspect.cleandoc(obj._doc or obj._description or '')])), self.wrapper)
    if itemdoc:
        self.out.write(itemdoc + '\n')