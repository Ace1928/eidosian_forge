import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def _tan(attrs, inputs, proto_obj):
    """Elementwise tan of input array."""
    return ('tan', attrs, inputs)