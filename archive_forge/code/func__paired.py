from django.db import NotSupportedError
from django.db.models.expressions import Func, Value
from django.db.models.fields import CharField, IntegerField, TextField
from django.db.models.functions import Cast, Coalesce
from django.db.models.lookups import Transform
def _paired(self, expressions):
    if len(expressions) == 2:
        return ConcatPair(*expressions)
    return ConcatPair(expressions[0], self._paired(expressions[1:]))