import datetime
import json
from django.contrib.postgres import forms, lookups
from django.db import models
from django.db.backends.postgresql.psycopg_any import (
from django.db.models.functions import Cast
from django.db.models.lookups import PostgresOperatorLookup
from .utils import AttributeSetter
class BigIntegerRangeField(RangeField):
    base_field = models.BigIntegerField
    range_type = NumericRange
    form_field = forms.IntegerRangeField

    def db_type(self, connection):
        return 'int8range'