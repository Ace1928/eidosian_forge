import inspect
from yaql.language import exceptions
from yaql.language import utils
from yaql.language import yaqltypes
def _parameter(name, value_type=None, nullable=None, alias=None):

    def wrapper(func):
        fd = _get_function_definition(func)
        fd.set_parameter(name, value_type, nullable, alias)
        return func
    return wrapper