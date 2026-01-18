import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class StringConstant(Constant):
    __slots__ = tuple()

    def __init__(self, nullable=False):
        super(StringConstant, self).__init__(nullable)

    def check(self, value, context, *args, **kwargs):
        return super(StringConstant, self).check(value, context, *args, **kwargs) and (value is None or isinstance(value.value, str))