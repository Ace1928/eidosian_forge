import numpy as np
from patsy import PatsyError
from patsy.util import (safe_isnan, safe_scalar_isnan,
def _desc_options(options):
    return ', '.join([repr(opt) for opt in options])