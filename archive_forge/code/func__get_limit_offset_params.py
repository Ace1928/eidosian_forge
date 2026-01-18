import datetime
import decimal
import json
import warnings
from importlib import import_module
import sqlparse
from django.conf import settings
from django.db import NotSupportedError, transaction
from django.db.backends import utils
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.encoding import force_str
def _get_limit_offset_params(self, low_mark, high_mark):
    offset = low_mark or 0
    if high_mark is not None:
        return (high_mark - offset, offset)
    elif offset:
        return (self.connection.ops.no_limit_value(), offset)
    return (None, offset)