import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class BooleanConstant(Constant):
    __slots__ = tuple()

    def __init__(self, nullable=False, expand=True):
        super(BooleanConstant, self).__init__(nullable, expand)

    def check(self, value, context, *args, **kwargs):
        return super(BooleanConstant, self).check(value, context, *args, **kwargs) and (value is None or type(value.value) is bool)