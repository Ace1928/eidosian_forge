from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField
class RowNumber(Func):
    function = 'ROW_NUMBER'
    output_field = IntegerField()
    window_compatible = True