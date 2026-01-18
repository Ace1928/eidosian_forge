import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def _prelu(attrs, inputs, proto_obj):
    """PRelu function"""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type': 'prelu'})
    return ('LeakyReLU', new_attrs, inputs)