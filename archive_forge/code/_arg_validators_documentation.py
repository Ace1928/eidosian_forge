import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings

    Check that all `fig` and `ax` kwargs are either ALL None, 
    or ALL are valid instances of Figures/Axes:
 
    An external Axes object can be passed in three places:
        - mpf.plot() `ax=` kwarg
        - mpf.plot() `volume=` kwarg
        - mpf.make_addplot() `ax=` kwarg
    ALL three places MUST be an Axes object, OR
    ALL three places MUST be None.  But it may not be mixed.
    