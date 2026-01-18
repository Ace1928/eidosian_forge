from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField
class DenseRank(Func):
    function = 'DENSE_RANK'
    output_field = IntegerField()
    window_compatible = True