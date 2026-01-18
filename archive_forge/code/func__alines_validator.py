import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _alines_validator(value, returnStandardizedValue=False):
    """
    Value for segments to be passed into LineCollection constructor must be:
    - a sequence of `lines`, where
    - a `lines` is a sequence of 2 or more vertices, where
    - a vertex is a `pair`, aka a sequence of two values, an x and a y point.

    From matplotlib.collections:
        `segments` are:
        A sequence of (line0, line1, line2), where:

        linen = (x0, y0), (x1, y1), ... (xm, ym)
       
        or the equivalent numpy array with two columns. Each line can be a different length.

    The above is from the matplotlib LineCollection documentation.
    It basically says that the "segments" passed into the LineCollection constructor 
    must be a Sequence of Sequences of 2 or more xy Pairs.  However here in `mplfinance`
    we want to allow that (seq of seq of xy pairs) _as well as_ just a sequence of pairs.
    Therefore here in the validator we will allow both:
       (a) seq of at least 2 date,float pairs         (this is a 'line'    as defined above)
       (b) seq of seqs of at least 2 date,float pairs (this is a 'seqment' as defined above)
    """
    if isinstance(value, dict):
        if 'alines' in value:
            value = value['alines']
        else:
            return False
    if not isinstance(value, (list, tuple)):
        return False if not returnStandardizedValue else None
    if not all([isinstance(line, (list, tuple)) and len(line) > 1 for line in value]):
        return False if not returnStandardizedValue else None
    if all([isinstance(point, (list, tuple)) and len(point) == 2 and _is_datelike(point[0]) and isinstance(point[1], (float, int)) for line in value for point in line]):
        return True if not returnStandardizedValue else value
    if all([isinstance(point, (list, tuple)) and len(point) == 2 and _is_datelike(point[0]) and isinstance(point[1], (float, int)) for point in value]):
        return True if not returnStandardizedValue else [value]
    return False if not returnStandardizedValue else None