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
class NullBooleanSelect(Select):
    """
    A Select Widget intended to be used with NullBooleanField.
    """

    def __init__(self, attrs=None):
        choices = (('unknown', _('Unknown')), ('true', _('Yes')), ('false', _('No')))
        super().__init__(attrs, choices)

    def format_value(self, value):
        try:
            return {True: 'true', False: 'false', 'true': 'true', 'false': 'false', '2': 'true', '3': 'false'}[value]
        except KeyError:
            return 'unknown'

    def value_from_datadict(self, data, files, name):
        value = data.get(name)
        return {True: True, 'True': True, 'False': False, False: False, 'true': True, 'false': False, '2': True, '3': False}.get(value)