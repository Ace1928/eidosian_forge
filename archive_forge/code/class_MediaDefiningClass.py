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
class MediaDefiningClass(type):
    """
    Metaclass for classes that can have media definitions.
    """

    def __new__(mcs, name, bases, attrs):
        new_class = super().__new__(mcs, name, bases, attrs)
        if 'media' not in attrs:
            new_class.media = media_property(new_class)
        return new_class