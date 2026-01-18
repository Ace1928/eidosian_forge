import functools
import gzip
import re
from difflib import SequenceMatcher
from pathlib import Path
from django.conf import settings
from django.core.exceptions import (
from django.utils.functional import cached_property, lazy
from django.utils.html import format_html, format_html_join
from django.utils.module_loading import import_string
from django.utils.translation import gettext as _
from django.utils.translation import ngettext
class NumericPasswordValidator:
    """
    Validate that the password is not entirely numeric.
    """

    def validate(self, password, user=None):
        if password.isdigit():
            raise ValidationError(_('This password is entirely numeric.'), code='password_entirely_numeric')

    def get_help_text(self):
        return _('Your password can’t be entirely numeric.')