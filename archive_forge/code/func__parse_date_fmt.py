import copy
import datetime
import warnings
from collections import defaultdict
from graphlib import CycleError, TopologicalSorter
from itertools import chain
from django.forms.utils import to_current_timezone
from django.templatetags.static import static
from django.utils import formats
from django.utils.choices import normalize_choices
from django.utils.dates import MONTHS
from django.utils.formats import get_format
from django.utils.html import format_html, html_safe
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
from .renderers import get_default_renderer
@staticmethod
def _parse_date_fmt():
    fmt = get_format('DATE_FORMAT')
    escaped = False
    for char in fmt:
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif char in 'Yy':
            yield 'year'
        elif char in 'bEFMmNn':
            yield 'month'
        elif char in 'dj':
            yield 'day'