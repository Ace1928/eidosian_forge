from .. import select
from .. import utils
from .._lazyload import matplotlib as mpl
from . import colors
from .tools import create_colormap
from .tools import create_normalize
from .tools import generate_colorbar
from .tools import generate_legend
from .tools import label_axis
from .utils import _get_figure
from .utils import _in_ipynb
from .utils import _is_color_array
from .utils import _with_default
from .utils import parse_fontsize
from .utils import show
from .utils import temp_fontsize
import numbers
import numpy as np
import pandas as pd
import warnings
def _label(self, label, values, idx):
    if label is False:
        return None
    elif label is not None:
        return label
    elif self._label_prefix is not None:
        return self._label_prefix + str(idx)
    elif label is not False and isinstance(values, pd.Series):
        return values.name
    else:
        return None