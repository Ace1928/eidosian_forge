from pythran.conversion import to_ast
from pythran.interval import UNKNOWN_RANGE, bool_values
from pythran.types.signature import extract_combiner
from pythran.typing import Any, Union, Fun, Generator
import gast as ast
class AttributeIntr(Intrinsic):
    """
    Internal representation for any attributes.

    Examples
    --------
    >> a.real
    """

    def __init__(self, **kwargs):
        """ Forward arguments. """
        super(AttributeIntr, self).__init__(**kwargs)
        if 'signature' in kwargs:
            self.signature = kwargs['signature']
        else:
            self.signature = Any

    def isattribute(self):
        """ Mark this intrinsic as an attribute. """
        return True