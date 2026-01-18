import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _valid_panel_id(panid):
    return panid in ['main', 'lower'] or (isinstance(panid, int) and panid >= 0 and (panid < 32))