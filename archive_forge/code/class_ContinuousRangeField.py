import datetime
import json
from django.contrib.postgres import forms, lookups
from django.db import models
from django.db.backends.postgresql.psycopg_any import (
from django.db.models.functions import Cast
from django.db.models.lookups import PostgresOperatorLookup
from .utils import AttributeSetter
class ContinuousRangeField(RangeField):
    """
    Continuous range field. It allows specifying default bounds for list and
    tuple inputs.
    """

    def __init__(self, *args, default_bounds=CANONICAL_RANGE_BOUNDS, **kwargs):
        if default_bounds not in ('[)', '(]', '()', '[]'):
            raise ValueError("default_bounds must be one of '[)', '(]', '()', or '[]'.")
        self.default_bounds = default_bounds
        super().__init__(*args, **kwargs)

    def get_prep_value(self, value):
        if isinstance(value, (list, tuple)):
            return self.range_type(value[0], value[1], self.default_bounds)
        return super().get_prep_value(value)

    def formfield(self, **kwargs):
        kwargs.setdefault('default_bounds', self.default_bounds)
        return super().formfield(**kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.default_bounds and self.default_bounds != CANONICAL_RANGE_BOUNDS:
            kwargs['default_bounds'] = self.default_bounds
        return (name, path, args, kwargs)