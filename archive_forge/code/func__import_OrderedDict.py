from __future__ import absolute_import
from decimal import Decimal
from .errors import JSONDecodeError
from .raw_json import RawJSON
from .decoder import JSONDecoder
from .encoder import JSONEncoder, JSONEncoderForHTML
def _import_OrderedDict():
    import collections
    try:
        return collections.OrderedDict
    except AttributeError:
        from . import ordered_dict
        return ordered_dict.OrderedDict