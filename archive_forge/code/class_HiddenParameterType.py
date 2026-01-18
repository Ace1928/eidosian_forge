import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class HiddenParameterType(metaclass=abc.ABCMeta):
    __slots__ = tuple()

    def check(self, value, context, engine, *args, **kwargs):
        return True