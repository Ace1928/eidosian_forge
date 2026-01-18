import itertools
import re
from collections import OrderedDict
from collections.abc import Generator
from typing import List
import numpy as np
import scipy as sp
from ..externals import _arff
from ..externals._arff import ArffSparseDataType
from ..utils import (
from ..utils.fixes import pd_fillna
def _pandas_arff_parser(gzip_file, output_arrays_type, openml_columns_info, feature_names_to_select, target_names_to_select, read_csv_kwargs=None):
    """ARFF parser using `pandas.read_csv`.

    This parser uses the metadata fetched directly from OpenML and skips the metadata
    headers of ARFF file itself. The data is loaded as a CSV file.

    Parameters
    ----------
    gzip_file : GzipFile instance
        The GZip compressed file with the ARFF formatted payload.

    output_arrays_type : {"numpy", "sparse", "pandas"}
        The type of the arrays that will be returned. The possibilities are:

        - `"numpy"`: both `X` and `y` will be NumPy arrays;
        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
          pandas Series or DataFrame.

    openml_columns_info : dict
        The information provided by OpenML regarding the columns of the ARFF
        file.

    feature_names_to_select : list of str
        A list of the feature names to be selected to build `X`.

    target_names_to_select : list of str
        A list of the target names to be selected to build `y`.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv`. It allows to overwrite
        the default options.

    Returns
    -------
    X : {ndarray, sparse matrix, dataframe}
        The data matrix.

    y : {ndarray, dataframe, series}
        The target.

    frame : dataframe or None
        A dataframe containing both `X` and `y`. `None` if
        `output_array_type != "pandas"`.

    categories : list of str or None
        The names of the features that are categorical. `None` if
        `output_array_type == "pandas"`.
    """
    import pandas as pd
    for line in gzip_file:
        if line.decode('utf-8').lower().startswith('@data'):
            break
    dtypes = {}
    for name in openml_columns_info:
        column_dtype = openml_columns_info[name]['data_type']
        if column_dtype.lower() == 'integer':
            dtypes[name] = 'Int64'
        elif column_dtype.lower() == 'nominal':
            dtypes[name] = 'category'
    dtypes_positional = {col_idx: dtypes[name] for col_idx, name in enumerate(openml_columns_info) if name in dtypes}
    default_read_csv_kwargs = {'header': None, 'index_col': False, 'na_values': ['?'], 'keep_default_na': False, 'comment': '%', 'quotechar': '"', 'skipinitialspace': True, 'escapechar': '\\', 'dtype': dtypes_positional}
    read_csv_kwargs = {**default_read_csv_kwargs, **(read_csv_kwargs or {})}
    frame = pd.read_csv(gzip_file, **read_csv_kwargs)
    try:
        frame.columns = [name for name in openml_columns_info]
    except ValueError as exc:
        raise pd.errors.ParserError('The number of columns provided by OpenML does not match the number of columns inferred by pandas when reading the file.') from exc
    columns_to_select = feature_names_to_select + target_names_to_select
    columns_to_keep = [col for col in frame.columns if col in columns_to_select]
    frame = frame[columns_to_keep]
    single_quote_pattern = re.compile("^'(?P<contents>.*)'$")

    def strip_single_quotes(input_string):
        match = re.search(single_quote_pattern, input_string)
        if match is None:
            return input_string
        return match.group('contents')
    categorical_columns = [name for name, dtype in frame.dtypes.items() if isinstance(dtype, pd.CategoricalDtype)]
    for col in categorical_columns:
        frame[col] = frame[col].cat.rename_categories(strip_single_quotes)
    X, y = _post_process_frame(frame, feature_names_to_select, target_names_to_select)
    if output_arrays_type == 'pandas':
        return (X, y, frame, None)
    else:
        X, y = (X.to_numpy(), y.to_numpy())
    categories = {name: dtype.categories.tolist() for name, dtype in frame.dtypes.items() if isinstance(dtype, pd.CategoricalDtype)}
    return (X, y, None, categories)