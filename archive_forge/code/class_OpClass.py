from django.db import NotSupportedError
from django.db.models import Func, Index
from django.utils.functional import cached_property
class OpClass(Func):
    template = '%(expressions)s %(name)s'

    def __init__(self, expression, name):
        super().__init__(expression, name=name)