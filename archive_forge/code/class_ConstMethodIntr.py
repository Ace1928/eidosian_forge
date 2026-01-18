from pythran.conversion import to_ast
from pythran.interval import UNKNOWN_RANGE, bool_values
from pythran.types.signature import extract_combiner
from pythran.typing import Any, Union, Fun, Generator
import gast as ast
class ConstMethodIntr(MethodIntr):

    def __init__(self, *combiners, **kwargs):
        kwargs.setdefault('argument_effects', (ReadEffect(),) * DefaultArgNum)
        super(ConstMethodIntr, self).__init__(*combiners, **kwargs)