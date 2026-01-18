import re
import collections
from . import _compat, tools
@classmethod
def _subcls(cls, other):
    name = '%s_%s' % (cls.__name__, other.__name__)
    bases = (other, cls)
    ns = {'__doc__': cls._doc % other.__name__}
    return type(name, bases, ns)