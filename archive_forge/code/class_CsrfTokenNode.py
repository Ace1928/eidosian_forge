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
class CsrfTokenNode(Node):
    child_nodelists = ()

    def render(self, context):
        csrf_token = context.get('csrf_token')
        if csrf_token:
            if csrf_token == 'NOTPROVIDED':
                return format_html('')
            else:
                return format_html('<input type="hidden" name="csrfmiddlewaretoken" value="{}">', csrf_token)
        else:
            if settings.DEBUG:
                warnings.warn('A {% csrf_token %} was used in a template, but the context did not provide the value.  This is usually caused by not using RequestContext.')
            return ''