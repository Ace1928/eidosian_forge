import datetime
import matplotlib.dates  as mdates
import matplotlib.colors as mcolors
import numpy as np
def _num_or_seq_of_num(value):
    return isinstance(value, (int, float, np.integer, np.floating)) or (isinstance(value, (list, tuple, np.ndarray)) and all([isinstance(v, (int, float, np.integer, np.floating)) for v in value]))