from django.db.models import Aggregate, FloatField, IntegerField
class Corr(StatAggregate):
    function = 'CORR'