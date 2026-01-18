import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
@staticmethod
def _find_function_context(spec, context):
    while context is not None:
        if spec in context:
            return context
        context = context.parent
    raise exceptions.NoFunctionRegisteredException(spec.name)