from django.db import NotSupportedError
from django.db.models.expressions import Func, Value
from django.db.models.fields import TextField
from django.db.models.fields.json import JSONField
from django.utils.regex_helper import _lazy_re_compile
class Coalesce(Func):
    """Return, from left to right, the first non-null expression."""
    function = 'COALESCE'

    def __init__(self, *expressions, **extra):
        if len(expressions) < 2:
            raise ValueError('Coalesce must take at least two expressions')
        super().__init__(*expressions, **extra)

    @property
    def empty_result_set_value(self):
        for expression in self.get_source_expressions():
            result = expression.empty_result_set_value
            if result is NotImplemented or result is not None:
                return result
        return None

    def as_oracle(self, compiler, connection, **extra_context):
        if self.output_field.get_internal_type() == 'TextField':
            clone = self.copy()
            clone.set_source_expressions([Func(expression, function='TO_NCLOB') for expression in self.get_source_expressions()])
            return super(Coalesce, clone).as_sql(compiler, connection, **extra_context)
        return self.as_sql(compiler, connection, **extra_context)