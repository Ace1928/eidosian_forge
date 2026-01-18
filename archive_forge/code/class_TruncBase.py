from datetime import datetime
from django.conf import settings
from django.db.models.expressions import Func
from django.db.models.fields import (
from django.db.models.lookups import (
from django.utils import timezone
class TruncBase(TimezoneMixin, Transform):
    kind = None
    tzinfo = None

    def __init__(self, expression, output_field=None, tzinfo=None, **extra):
        self.tzinfo = tzinfo
        super().__init__(expression, output_field=output_field, **extra)

    def as_sql(self, compiler, connection):
        sql, params = compiler.compile(self.lhs)
        tzname = None
        if isinstance(self.lhs.output_field, DateTimeField):
            tzname = self.get_tzname()
        elif self.tzinfo is not None:
            raise ValueError('tzinfo can only be used with DateTimeField.')
        if isinstance(self.output_field, DateTimeField):
            sql, params = connection.ops.datetime_trunc_sql(self.kind, sql, tuple(params), tzname)
        elif isinstance(self.output_field, DateField):
            sql, params = connection.ops.date_trunc_sql(self.kind, sql, tuple(params), tzname)
        elif isinstance(self.output_field, TimeField):
            sql, params = connection.ops.time_trunc_sql(self.kind, sql, tuple(params), tzname)
        else:
            raise ValueError('Trunc only valid on DateField, TimeField, or DateTimeField.')
        return (sql, params)

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        copy = super().resolve_expression(query, allow_joins, reuse, summarize, for_save)
        field = copy.lhs.output_field
        if not isinstance(field, (DateField, TimeField)):
            raise TypeError("%r isn't a DateField, TimeField, or DateTimeField." % field.name)
        if not isinstance(copy.output_field, (DateField, DateTimeField, TimeField)):
            raise ValueError('output_field must be either DateField, TimeField, or DateTimeField')
        class_output_field = self.__class__.output_field if isinstance(self.__class__.output_field, Field) else None
        output_field = class_output_field or copy.output_field
        has_explicit_output_field = class_output_field or field.__class__ is not copy.output_field.__class__
        if type(field) is DateField and (isinstance(output_field, DateTimeField) or copy.kind in ('hour', 'minute', 'second', 'time')):
            raise ValueError("Cannot truncate DateField '%s' to %s." % (field.name, output_field.__class__.__name__ if has_explicit_output_field else 'DateTimeField'))
        elif isinstance(field, TimeField) and (isinstance(output_field, DateTimeField) or copy.kind in ('year', 'quarter', 'month', 'week', 'day', 'date')):
            raise ValueError("Cannot truncate TimeField '%s' to %s." % (field.name, output_field.__class__.__name__ if has_explicit_output_field else 'DateTimeField'))
        return copy

    def convert_value(self, value, expression, connection):
        if isinstance(self.output_field, DateTimeField):
            if not settings.USE_TZ:
                pass
            elif value is not None:
                value = value.replace(tzinfo=None)
                value = timezone.make_aware(value, self.tzinfo)
            elif not connection.features.has_zoneinfo_database:
                raise ValueError('Database returned an invalid datetime value. Are time zone definitions for your database installed?')
        elif isinstance(value, datetime):
            if value is None:
                pass
            elif isinstance(self.output_field, DateField):
                value = value.date()
            elif isinstance(self.output_field, TimeField):
                value = value.time()
        return value