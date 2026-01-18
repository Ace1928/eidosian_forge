from .. import utils
from .utils import _matrix_to_data_frame
import pandas as pd
def _read_csv_sparse(filename, chunksize=10000, fill_value=0.0, **kwargs):
    """Read a csv file into a pd.DataFrame[pd.SparseArray]."""
    chunks = pd.read_csv(filename, chunksize=chunksize, **kwargs)
    data = pd.concat((utils.dataframe_to_sparse(chunk, fill_value=fill_value) for chunk in chunks))
    return data