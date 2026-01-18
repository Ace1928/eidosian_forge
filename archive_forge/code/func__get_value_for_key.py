from calendar import timegm
from decimal import Decimal as MyDecimal, ROUND_HALF_EVEN
from email.utils import formatdate
import six
from flask_restful import marshal
from flask import url_for, request
def _get_value_for_key(key, obj, default):
    if is_indexable_but_not_string(obj):
        try:
            return obj[key]
        except (IndexError, TypeError, KeyError):
            pass
    return getattr(obj, key, default)