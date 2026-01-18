from . import hdf5
from .utils import _matrix_to_data_frame
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import shutil
import tempfile
import urllib
import warnings
import zipfile
def _combine_gene_id(symbols, ids):
    """Create gene labels of the form SYMBOL (ID).

    Parameters
    ----------
    genes: pandas.DataFrame with columns['symbol', 'id']

    Returns
    -------
    pandas.Index with combined gene symbols and ids
    """
    columns = np.core.defchararray.add(np.array(symbols, dtype=str), ' (')
    columns = np.core.defchararray.add(columns, np.array(ids, dtype=str))
    columns = np.core.defchararray.add(columns, ')')
    return columns