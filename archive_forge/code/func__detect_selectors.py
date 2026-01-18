import os
import re
import sys
import traceback
import ast
import importlib
from re import sub, findall
from types import CodeType
from functools import partial
from collections import OrderedDict, defaultdict
import kivy.lang.builder  # imported as absolute to avoid circular import
from kivy.logger import Logger
from kivy.cache import Cache
from kivy import require
from kivy.resources import resource_find
from kivy.utils import rgba
import kivy.metrics as Metrics
def _detect_selectors(self):
    c = self.name[0]
    if c == '<':
        self._build_rule()
    elif c == '[':
        self._build_template()
    else:
        if self.ctx.root is not None:
            raise ParserException(self.ctx, self.line, 'Only one root object is allowed by .kv')
        self.ctx.root = self