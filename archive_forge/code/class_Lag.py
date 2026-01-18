from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField
class Lag(LagLeadFunction):
    function = 'LAG'