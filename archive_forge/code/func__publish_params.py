import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
@staticmethod
def _publish_params(context, args, kwargs):
    for i, param in enumerate(args):
        context['$' + str(i + 1)] = param
    for arg_name, arg_value in kwargs.items():
        context['$' + arg_name] = arg_value