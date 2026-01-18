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
def _parse_10x_genes(symbols, ids, gene_labels='symbol', allow_duplicates=True):
    assert gene_labels in ['symbol', 'id', 'both']
    if gene_labels == 'symbol':
        columns = symbols
        if not allow_duplicates and len(np.unique(columns)) < len(columns):
            warnings.warn("Duplicate gene names detected! Forcing `gene_labels='both'`. Alternatively, try `gene_labels='id'`, `allow_duplicates=True`, or load the matrix with `sparse=False`", RuntimeWarning)
            gene_labels = 'both'
    if gene_labels == 'both':
        columns = _combine_gene_id(symbols, ids)
    elif gene_labels == 'id':
        columns = ids
    return columns