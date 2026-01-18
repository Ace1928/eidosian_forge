import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
def _check_match(self, value, context, engine, *args, **kwargs):
    for type_to_check in self.types:
        check_result = type_to_check.check(value, context, engine, *args, **kwargs)
        if check_result:
            return type_to_check