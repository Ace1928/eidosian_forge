from pythran.conversion import to_ast
from pythran.interval import UNKNOWN_RANGE, bool_values
from pythran.types.signature import extract_combiner
from pythran.typing import Any, Union, Fun, Generator
import gast as ast
class ConstFunctionIntr(FunctionIntr):

    def __init__(self, **kwargs):
        kwargs.setdefault('argument_effects', (ReadEffect(),) * DefaultArgNum)
        super(ConstFunctionIntr, self).__init__(**kwargs)