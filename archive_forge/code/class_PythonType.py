import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class PythonType(GenericType):
    __slots__ = ('python_type', 'validators')

    def __init__(self, python_type, nullable=True, validators=None):
        self.python_type = python_type
        if not validators:
            validators = [lambda _: True]
        if not isinstance(validators, (list, tuple)):
            validators = [validators]
        self.validators = validators
        super(PythonType, self).__init__(nullable, lambda value, context, *args, **kwargs: isinstance(value, self.python_type) and all(map(lambda t: t(value), self.validators)))

    def is_specialization_of(self, other):
        if not isinstance(other, PythonType):
            return False
        try:
            len(self.python_type)
            len(other.python_type)
        except Exception:
            return issubclass(self.python_type, other.python_type) and (not issubclass(other.python_type, self.python_type))
        else:
            return False