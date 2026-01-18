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
class TemplateTagNode(Node):
    mapping = {'openblock': BLOCK_TAG_START, 'closeblock': BLOCK_TAG_END, 'openvariable': VARIABLE_TAG_START, 'closevariable': VARIABLE_TAG_END, 'openbrace': SINGLE_BRACE_START, 'closebrace': SINGLE_BRACE_END, 'opencomment': COMMENT_TAG_START, 'closecomment': COMMENT_TAG_END}

    def __init__(self, tagtype):
        self.tagtype = tagtype

    def render(self, context):
        return self.mapping.get(self.tagtype, '')