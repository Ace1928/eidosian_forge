import datetime
import json
from django.contrib.postgres import forms, lookups
from django.db import models
from django.db.backends.postgresql.psycopg_any import (
from django.db.models.functions import Cast
from django.db.models.lookups import PostgresOperatorLookup
from .utils import AttributeSetter
@RangeField.register_lookup
class NotGreaterThan(PostgresOperatorLookup):
    lookup_name = 'not_gt'
    postgres_operator = RangeOperators.NOT_GT