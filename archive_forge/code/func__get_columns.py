from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _get_columns(data):
    return data.columns if isinstance(data, pd.DataFrame) else data.index