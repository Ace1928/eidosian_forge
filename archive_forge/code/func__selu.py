import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def _selu(attrs, inputs, proto_obj):
    """Selu function"""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type': 'selu'})
    return ('LeakyReLU', new_attrs, inputs)