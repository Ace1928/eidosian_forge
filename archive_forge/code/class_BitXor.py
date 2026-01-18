import json
import warnings
from django.contrib.postgres.fields import ArrayField
from django.db.models import Aggregate, BooleanField, JSONField, TextField, Value
from django.utils.deprecation import RemovedInDjango51Warning
from .mixins import OrderableAggMixin
class BitXor(Aggregate):
    function = 'BIT_XOR'