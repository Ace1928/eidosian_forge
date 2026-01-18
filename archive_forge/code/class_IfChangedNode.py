import re
import sys
import warnings
from collections import namedtuple
from datetime import datetime
from itertools import cycle as itertools_cycle
from itertools import groupby
from django.conf import settings
from django.utils import timezone
from django.utils.html import conditional_escape, escape, format_html
from django.utils.lorem_ipsum import paragraphs, words
from django.utils.safestring import mark_safe
from .base import (
from .context import Context
from .defaultfilters import date
from .library import Library
from .smartif import IfParser, Literal
class IfChangedNode(Node):
    child_nodelists = ('nodelist_true', 'nodelist_false')

    def __init__(self, nodelist_true, nodelist_false, *varlist):
        self.nodelist_true = nodelist_true
        self.nodelist_false = nodelist_false
        self._varlist = varlist

    def render(self, context):
        state_frame = self._get_context_stack_frame(context)
        state_frame.setdefault(self)
        nodelist_true_output = None
        if self._varlist:
            compare_to = [var.resolve(context, ignore_failures=True) for var in self._varlist]
        else:
            compare_to = nodelist_true_output = self.nodelist_true.render(context)
        if compare_to != state_frame[self]:
            state_frame[self] = compare_to
            return nodelist_true_output or self.nodelist_true.render(context)
        elif self.nodelist_false:
            return self.nodelist_false.render(context)
        return ''

    def _get_context_stack_frame(self, context):
        if 'forloop' in context:
            return context['forloop']
        else:
            return context.render_context