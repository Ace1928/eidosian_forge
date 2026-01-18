import matplotlib.pyplot as plt
import copy
import pprint
import os.path as path
from   mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from   mplfinance._styledata      import _styles
from   mplfinance._helpers        import _mpf_is_color_like
def _valid_mpf_color_spec(value):
    """value must be a color, "inherit"-like, or dict of colors"""
    return _mpf_is_color_like(value) or (isinstance(value, str) and value == 'inherit'[0:len(value)]) or (isinstance(value, dict) and all([_mpf_is_color_like(v) for v in value.values()]))