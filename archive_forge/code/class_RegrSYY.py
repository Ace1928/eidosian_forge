from django.db.models import Aggregate, FloatField, IntegerField
class RegrSYY(StatAggregate):
    function = 'REGR_SYY'