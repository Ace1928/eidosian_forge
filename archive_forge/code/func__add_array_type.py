from numpy.compat import unicode
from numpy.core._string_helpers import english_lower
from numpy.core.multiarray import typeinfo, dtype
from numpy.core._dtype import _kind_name
def _add_array_type(typename, bits):
    try:
        t = allTypes['%s%d' % (typename, bits)]
    except KeyError:
        pass
    else:
        sctypes[typename].append(t)