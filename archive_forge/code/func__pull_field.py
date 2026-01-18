import copy
from collections import defaultdict
import numpy as np
from pandas import compat, DataFrame
def _pull_field(js, spec):
    result = js
    if isinstance(spec, list):
        for field in spec:
            result = result[field]
    else:
        result = result[spec]
    return result