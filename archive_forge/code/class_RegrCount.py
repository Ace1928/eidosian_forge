from django.db.models import Aggregate, FloatField, IntegerField
class RegrCount(StatAggregate):
    function = 'REGR_COUNT'
    output_field = IntegerField()
    empty_result_set_value = 0