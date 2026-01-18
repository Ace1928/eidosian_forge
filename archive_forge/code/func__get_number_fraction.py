import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
def _get_number_fraction(self, factor):
    number_fraction = None
    for threshold in [1, 60, 3600]:
        if factor <= threshold:
            break
        d = factor // threshold
        int_log_d = int(np.floor(np.log10(d)))
        if 10 ** int_log_d == d and d != 1:
            number_fraction = int_log_d
            factor = factor // 10 ** int_log_d
            return (factor, number_fraction)
    return (factor, number_fraction)