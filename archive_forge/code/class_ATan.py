import math
from django.db.models.expressions import Func, Value
from django.db.models.fields import FloatField, IntegerField
from django.db.models.functions import Cast
from django.db.models.functions.mixins import (
from django.db.models.lookups import Transform
class ATan(NumericOutputFieldMixin, Transform):
    function = 'ATAN'
    lookup_name = 'atan'