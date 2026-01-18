import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class LazyParameterType(metaclass=abc.ABCMeta):
    __slots__ = tuple()