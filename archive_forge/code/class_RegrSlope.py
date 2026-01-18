from django.db.models import Aggregate, FloatField, IntegerField
class RegrSlope(StatAggregate):
    function = 'REGR_SLOPE'