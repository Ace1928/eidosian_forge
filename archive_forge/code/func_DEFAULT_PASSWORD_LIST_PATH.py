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
@cached_property
def DEFAULT_PASSWORD_LIST_PATH(self):
    return Path(__file__).resolve().parent / 'common-passwords.txt.gz'