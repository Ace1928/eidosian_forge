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
class ParserRule(object):
    """Represents a rule, in terms of the Kivy internal language.
    """
    __slots__ = ('ctx', 'line', 'name', 'children', 'id', 'properties', 'canvas_before', 'canvas_root', 'canvas_after', 'handlers', 'level', 'cache_marked', 'avoid_previous_rules')

    def __init__(self, ctx, line, name, level):
        super(ParserRule, self).__init__()
        self.level = level
        self.ctx = ctx
        self.line = line
        self.name = name
        self.children = []
        self.id = None
        self.properties = OrderedDict()
        self.canvas_root = None
        self.canvas_before = None
        self.canvas_after = None
        self.handlers = []
        self.cache_marked = []
        self.avoid_previous_rules = False
        if level == 0:
            self._detect_selectors()
        else:
            self._forbid_selectors()

    def precompile(self):
        for x in self.properties.values():
            x.precompile()
        for x in self.handlers:
            x.precompile()
        for x in self.children:
            x.precompile()
        if self.canvas_before:
            self.canvas_before.precompile()
        if self.canvas_root:
            self.canvas_root.precompile()
        if self.canvas_after:
            self.canvas_after.precompile()

    def create_missing(self, widget):
        cls = widget.__class__
        if cls in self.cache_marked:
            return
        self.cache_marked.append(cls)
        for name in self.properties:
            if hasattr(widget, name):
                continue
            value = self.properties[name].co_value
            if type(value) is CodeType:
                value = None
            widget.create_property(name, value, default_value=False)

    def _forbid_selectors(self):
        c = self.name[0]
        if c == '<' or c == '[':
            raise ParserException(self.ctx, self.line, 'Selectors rules are allowed only at the first level')

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

    def _build_template(self):
        name = self.name
        exception = ParserException(self.ctx, self.line, 'Deprecated Kivy lang template syntax used "{}". Templates will be removed in a future version'.format(name))
        if name not in ('[FileListEntry@FloatLayout+TreeViewNode]', '[FileIconEntry@Widget]', '[AccordionItemTitle@Label]'):
            Logger.warning(exception)
        if __debug__:
            trace('Builder: build template for %s' % name)
        if name[0] != '[' or name[-1] != ']':
            raise ParserException(self.ctx, self.line, 'Invalid template (must be inside [])')
        item_content = name[1:-1]
        if '@' not in item_content:
            raise ParserException(self.ctx, self.line, 'Invalid template name (missing @)')
        template_name, template_root_cls = item_content.split('@')
        self.ctx.templates.append((template_name, template_root_cls, self))

    def __repr__(self):
        return '<ParserRule name=%r>' % (self.name,)