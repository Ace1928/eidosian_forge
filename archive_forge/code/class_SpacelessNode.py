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
class SpacelessNode(Node):

    def __init__(self, nodelist):
        self.nodelist = nodelist

    def render(self, context):
        from django.utils.html import strip_spaces_between_tags
        return strip_spaces_between_tags(self.nodelist.render(context).strip())