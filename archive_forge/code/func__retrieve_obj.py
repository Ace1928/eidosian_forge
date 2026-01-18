import posixpath
import re
from ._hl.attrs import AttributeManager
from ._hl.base import HLObject
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import generics
def _retrieve_obj(name, context):
    """ Filter function for completion. """
    if '(' in name:
        raise ValueError()
    return eval(name, context.user_ns)