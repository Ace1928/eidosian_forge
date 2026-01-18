import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
class Super(HiddenParameterType, SmartType):
    __slots__ = ('with_context', 'method', 'with_name')

    def __init__(self, with_context=False, method=None, with_name=False):
        self.with_context = with_context
        self.method = method
        self.with_name = with_name
        super(Super, self).__init__(False)

    @staticmethod
    def _find_function_context(spec, context):
        while context is not None:
            if spec in context:
                return context
            context = context.parent
        raise exceptions.NoFunctionRegisteredException(spec.name)

    def convert(self, value, receiver, context, function_spec, engine, *convert_args, **convert_kwargs):
        if callable(value) and hasattr(value, '__unwrapped__'):
            value = value.__unwrapped__

        def func(*args, **kwargs):
            function_context = self._find_function_context(function_spec, context)
            parent_function_context = function_context.parent
            if parent_function_context is None:
                raise exceptions.NoFunctionRegisteredException(function_spec.name)
            new_name = function_spec.name
            if self.with_name:
                new_name = args[0]
                args = args[1:]
            new_receiver = receiver
            if self.method is True:
                new_receiver = args[0]
                args = args[1:]
            elif self.method is False:
                new_receiver = utils.NO_VALUE
            if self.with_context:
                new_context = args[0]
                args = args[1:]
            else:
                new_context = context.create_child_context()
            return parent_function_context(new_name, engine, new_receiver, new_context)(*args, **kwargs)
        func.__unwrapped__ = value
        return func