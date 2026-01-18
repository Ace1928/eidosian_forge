import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def _elu(attrs, inputs, proto_obj):
    """Elu function"""
    if 'alpha' in attrs:
        new_attrs = translation_utils._fix_attribute_names(attrs, {'alpha': 'slope'})
    else:
        new_attrs = translation_utils._add_extra_attributes(attrs, {'slope': 1.0})
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'act_type': 'elu'})
    return ('LeakyReLU', new_attrs, inputs)