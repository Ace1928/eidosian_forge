from django.db.models import Aggregate, FloatField, IntegerField
class RegrAvgY(StatAggregate):
    function = 'REGR_AVGY'