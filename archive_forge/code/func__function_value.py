from inspect import Parameter
from jedi.cache import memoize_method
from jedi import debug
from jedi import parser_utils
@property
def _function_value(self):
    if self.__function_value is None:
        return self.value
    return self.__function_value