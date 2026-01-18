from .. import sanitize
from .. import utils
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
def _parse_header(header, n_expected, header_type='gene_names'):
    """Parse row or column names from a file.

    Parameters
    ----------
    header : `str` filename, array-like or `None`
    n_expected : `int`
        Expected header length
    header_type : str
        argument name for error printing

    Returns
    -------
    columns : list-like or `None`
        Parsed column names.
    """
    if header is None or header is False:
        return None
    elif isinstance(header, str):
        if header.endswith('tsv'):
            delimiter = '\t'
        else:
            delimiter = ','
        columns = pd.read_csv(header, delimiter=delimiter, header=None).values.flatten().astype(str)
        if not len(columns) == n_expected:
            raise ValueError('Expected {} entries in {}. Got {}'.format(n_expected, header, len(columns)))
    else:
        columns = header
        if not len(columns) == n_expected:
            raise ValueError('Expected {} entries in {}. Got {}'.format(n_expected, header_type, len(columns)))
    return columns