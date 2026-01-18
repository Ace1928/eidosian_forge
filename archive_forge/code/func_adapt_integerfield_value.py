import json
from functools import lru_cache, partial
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.postgresql.psycopg_any import (
from django.db.backends.utils import split_tzname_delta
from django.db.models.constants import OnConflict
from django.db.models.functions import Cast
from django.utils.regex_helper import _lazy_re_compile
def adapt_integerfield_value(self, value, internal_type):
    if value is None or hasattr(value, 'resolve_expression'):
        return value
    return self.integerfield_type_map[internal_type](value)