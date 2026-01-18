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
class WithNode(Node):

    def __init__(self, var, name, nodelist, extra_context=None):
        self.nodelist = nodelist
        self.extra_context = extra_context or {}
        if name:
            self.extra_context[name] = var

    def __repr__(self):
        return '<%s>' % self.__class__.__name__

    def render(self, context):
        values = {key: val.resolve(context) for key, val in self.extra_context.items()}
        with context.push(**values):
            return self.nodelist.render(context)