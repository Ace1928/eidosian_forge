from django.db.models import Aggregate, FloatField, IntegerField
class RegrSXY(StatAggregate):
    function = 'REGR_SXY'