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
class RadioSelect(ChoiceWidget):
    input_type = 'radio'
    template_name = 'django/forms/widgets/radio.html'
    option_template_name = 'django/forms/widgets/radio_option.html'
    use_fieldset = True

    def id_for_label(self, id_, index=None):
        """
        Don't include for="field_0" in <label> to improve accessibility when
        using a screen reader, in addition clicking such a label would toggle
        the first input.
        """
        if index is None:
            return ''
        return super().id_for_label(id_, index)