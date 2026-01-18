from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField
class FirstValue(Func):
    arity = 1
    function = 'FIRST_VALUE'
    window_compatible = True