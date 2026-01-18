import json
import warnings
from django.contrib.postgres.fields import ArrayField
from django.db.models import Aggregate, BooleanField, JSONField, TextField, Value
from django.utils.deprecation import RemovedInDjango51Warning
from .mixins import OrderableAggMixin
class ArrayAgg(OrderableAggMixin, Aggregate):
    function = 'ARRAY_AGG'
    template = '%(function)s(%(distinct)s%(expressions)s %(ordering)s)'
    allow_distinct = True

    @property
    def output_field(self):
        return ArrayField(self.source_expressions[0].output_field)