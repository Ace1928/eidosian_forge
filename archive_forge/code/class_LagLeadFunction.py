from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField
class LagLeadFunction(Func):
    window_compatible = True

    def __init__(self, expression, offset=1, default=None, **extra):
        if expression is None:
            raise ValueError('%s requires a non-null source expression.' % self.__class__.__name__)
        if offset is None or offset <= 0:
            raise ValueError('%s requires a positive integer for the offset.' % self.__class__.__name__)
        args = (expression, offset)
        if default is not None:
            args += (default,)
        super().__init__(*args, **extra)

    def _resolve_output_field(self):
        sources = self.get_source_expressions()
        return sources[0].output_field