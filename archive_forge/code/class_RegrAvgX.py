from django.db.models import Aggregate, FloatField, IntegerField
class RegrAvgX(StatAggregate):
    function = 'REGR_AVGX'