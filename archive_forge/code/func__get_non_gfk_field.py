import datetime
import decimal
import json
from collections import defaultdict
from functools import reduce
from operator import or_
from django.core.exceptions import FieldDoesNotExist
from django.db import models, router
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import Collector
from django.forms.utils import pretty_name
from django.urls import NoReverseMatch, reverse
from django.utils import formats, timezone
from django.utils.hashable import make_hashable
from django.utils.html import format_html
from django.utils.regex_helper import _lazy_re_compile
from django.utils.text import capfirst
from django.utils.translation import ngettext
from django.utils.translation import override as translation_override
def _get_non_gfk_field(opts, name):
    """
    For historical reasons, the admin app relies on GenericForeignKeys as being
    "not found" by get_field(). This could likely be cleaned up.

    Reverse relations should also be excluded as these aren't attributes of the
    model (rather something like `foo_set`).
    """
    field = opts.get_field(name)
    if field.is_relation and (field.many_to_one and (not field.related_model) or field.one_to_many):
        raise FieldDoesNotExist()
    if field.is_relation and (not field.many_to_many) and hasattr(field, 'attname') and (field.attname == name):
        raise FieldIsAForeignKeyColumnName()
    return field