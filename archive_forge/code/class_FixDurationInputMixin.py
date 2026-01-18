import sys
from django.db.models.fields import DecimalField, FloatField, IntegerField
from django.db.models.functions import Cast
class FixDurationInputMixin:

    def as_mysql(self, compiler, connection, **extra_context):
        sql, params = super().as_sql(compiler, connection, **extra_context)
        if self.output_field.get_internal_type() == 'DurationField':
            sql = 'CAST(%s AS SIGNED)' % sql
        return (sql, params)

    def as_oracle(self, compiler, connection, **extra_context):
        if self.output_field.get_internal_type() == 'DurationField':
            expression = self.get_source_expressions()[0]
            options = self._get_repr_options()
            from django.db.backends.oracle.functions import IntervalToSeconds, SecondsToInterval
            return compiler.compile(SecondsToInterval(self.__class__(IntervalToSeconds(expression), **options)))
        return super().as_sql(compiler, connection, **extra_context)