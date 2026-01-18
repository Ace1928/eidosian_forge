from pythran.conversion import to_ast
from pythran.interval import UNKNOWN_RANGE, bool_values
from pythran.types.signature import extract_combiner
from pythran.typing import Any, Union, Fun, Generator
import gast as ast
class ExceptionClass(Class, ConstExceptionIntr):

    def __init__(self, d, *args, **kwargs):
        super(ExceptionClass, self).__init__(d, *args, **kwargs)