from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField
class PercentRank(Func):
    function = 'PERCENT_RANK'
    output_field = FloatField()
    window_compatible = True