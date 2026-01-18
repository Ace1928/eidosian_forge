import abc
from yaql.language import exceptions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
@staticmethod
def _import_function_definition(fd):
    return fd