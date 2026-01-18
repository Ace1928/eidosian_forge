from django.db.models import DecimalField, DurationField, Func
class IntervalToSeconds(Func):
    function = ''
    template = '\n    EXTRACT(day from %(expressions)s) * 86400 +\n    EXTRACT(hour from %(expressions)s) * 3600 +\n    EXTRACT(minute from %(expressions)s) * 60 +\n    EXTRACT(second from %(expressions)s)\n    '

    def __init__(self, expression, *, output_field=None, **extra):
        super().__init__(expression, output_field=output_field or DecimalField(), **extra)