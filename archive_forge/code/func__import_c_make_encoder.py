from __future__ import absolute_import
from decimal import Decimal
from .errors import JSONDecodeError
from .raw_json import RawJSON
from .decoder import JSONDecoder
from .encoder import JSONEncoder, JSONEncoderForHTML
def _import_c_make_encoder():
    try:
        from ._speedups import make_encoder
        return make_encoder
    except ImportError:
        return None