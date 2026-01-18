from django.db.models import Aggregate, FloatField, IntegerField
class StatAggregate(Aggregate):
    output_field = FloatField()

    def __init__(self, y, x, output_field=None, filter=None, default=None):
        if not x or not y:
            raise ValueError('Both y and x must be provided.')
        super().__init__(y, x, output_field=output_field, filter=filter, default=default)