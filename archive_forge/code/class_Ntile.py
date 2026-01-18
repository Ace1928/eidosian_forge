from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField
class Ntile(Func):
    function = 'NTILE'
    output_field = IntegerField()
    window_compatible = True

    def __init__(self, num_buckets=1, **extra):
        if num_buckets <= 0:
            raise ValueError('num_buckets must be greater than 0.')
        super().__init__(num_buckets, **extra)