from .. import utils
from .._lazyload import fcsparser
from .utils import _matrix_to_data_frame
from io import BytesIO
import numpy as np
import pandas as pd
import string
import struct
import warnings
def _channel_names_from_meta(meta, channel_numbers, naming='N'):
    try:
        return tuple([meta['$P{0}{1}'.format(i, naming)] for i in channel_numbers])
    except KeyError:
        return []