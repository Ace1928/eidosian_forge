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
class FirstOfNode(Node):

    def __init__(self, variables, asvar=None):
        self.vars = variables
        self.asvar = asvar

    def render(self, context):
        first = ''
        for var in self.vars:
            value = var.resolve(context, ignore_failures=True)
            if value:
                first = render_value_in_context(value, context)
                break
        if self.asvar:
            context[self.asvar] = first
            return ''
        return first