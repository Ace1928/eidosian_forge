import sys
from django.db.models.fields import DecimalField, FloatField, IntegerField
from django.db.models.functions import Cast
class NumericOutputFieldMixin:

    def _resolve_output_field(self):
        source_fields = self.get_source_fields()
        if any((isinstance(s, DecimalField) for s in source_fields)):
            return DecimalField()
        if any((isinstance(s, IntegerField) for s in source_fields)):
            return FloatField()
        return super()._resolve_output_field() if source_fields else FloatField()