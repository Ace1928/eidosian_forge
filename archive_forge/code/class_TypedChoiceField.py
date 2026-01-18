import copy
import datetime
import json
import math
import operator
import os
import re
import uuid
import warnings
from decimal import Decimal, DecimalException
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit
from django.conf import settings
from django.core import validators
from django.core.exceptions import ValidationError
from django.forms.boundfield import BoundField
from django.forms.utils import from_current_timezone, to_current_timezone
from django.forms.widgets import (
from django.utils import formats
from django.utils.choices import normalize_choices
from django.utils.dateparse import parse_datetime, parse_duration
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.duration import duration_string
from django.utils.ipv6 import clean_ipv6_address
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
class TypedChoiceField(ChoiceField):

    def __init__(self, *, coerce=lambda val: val, empty_value='', **kwargs):
        self.coerce = coerce
        self.empty_value = empty_value
        super().__init__(**kwargs)

    def _coerce(self, value):
        """
        Validate that the value can be coerced to the right type (if not empty).
        """
        if value == self.empty_value or value in self.empty_values:
            return self.empty_value
        try:
            value = self.coerce(value)
        except (ValueError, TypeError, ValidationError):
            raise ValidationError(self.error_messages['invalid_choice'], code='invalid_choice', params={'value': value})
        return value

    def clean(self, value):
        value = super().clean(value)
        return self._coerce(value)