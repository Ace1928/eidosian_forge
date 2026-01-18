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
def _build_rule(self):
    name = self.name
    if __debug__:
        trace('Builder: build rule for %s' % name)
    if name[0] != '<' or name[-1] != '>':
        raise ParserException(self.ctx, self.line, 'Invalid rule (must be inside <>)')
    name = name[1:-1]
    if name[:1] == '-':
        self.avoid_previous_rules = True
        name = name[1:]
    for rule in re.split(lang_cls_split_pat, name):
        crule = None
        if not rule:
            raise ParserException(self.ctx, self.line, 'Empty rule detected')
        if '@' in rule:
            rule, baseclasses = rule.split('@', 1)
            if not re.match(lang_key, rule):
                raise ParserException(self.ctx, self.line, 'Invalid dynamic class name')
            self.ctx.dynamic_classes[rule] = baseclasses
            crule = ParserSelectorName(rule)
        elif rule[0] == '.':
            crule = ParserSelectorClass(rule[1:])
        else:
            crule = ParserSelectorName(rule)
        self.ctx.rules.append((crule, self))