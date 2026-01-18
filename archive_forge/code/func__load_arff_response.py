import gzip
import hashlib
import json
import os
import shutil
import time
from contextlib import closing
from functools import wraps
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from warnings import warn
import numpy as np
from ..utils import (
from ..utils._param_validation import (
from . import get_data_home
from ._arff_parser import load_arff_from_gzip_file
def _load_arff_response(url: str, data_home: Optional[str], parser: str, output_type: str, openml_columns_info: dict, feature_names_to_select: List[str], target_names_to_select: List[str], shape: Optional[Tuple[int, int]], md5_checksum: str, n_retries: int=3, delay: float=1.0, read_csv_kwargs: Optional[Dict]=None):
    """Load the ARFF data associated with the OpenML URL.

    In addition of loading the data, this function will also check the
    integrity of the downloaded file from OpenML using MD5 checksum.

    Parameters
    ----------
    url : str
        The URL of the ARFF file on OpenML.

    data_home : str
        The location where to cache the data.

    parser : {"liac-arff", "pandas"}
        The parser used to parse the ARFF file.

    output_type : {"numpy", "pandas", "sparse"}
        The type of the arrays that will be returned. The possibilities are:

        - `"numpy"`: both `X` and `y` will be NumPy arrays;
        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
          pandas Series or DataFrame.

    openml_columns_info : dict
        The information provided by OpenML regarding the columns of the ARFF
        file.

    feature_names_to_select : list of str
        The list of the features to be selected.

    target_names_to_select : list of str
        The list of the target variables to be selected.

    shape : tuple or None
        With `parser="liac-arff"`, when using a generator to load the data,
        one needs to provide the shape of the data beforehand.

    md5_checksum : str
        The MD5 checksum provided by OpenML to check the data integrity.

    n_retries : int, default=3
        The number of times to retry downloading the data if it fails.

    delay : float, default=1.0
        The delay between two consecutive downloads in seconds.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv` when using the pandas parser.
        It allows to overwrite the default options.

        .. versionadded:: 1.3

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
    gzip_file = _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
    with closing(gzip_file):
        md5 = hashlib.md5()
        for chunk in iter(lambda: gzip_file.read(4096), b''):
            md5.update(chunk)
        actual_md5_checksum = md5.hexdigest()
    if actual_md5_checksum != md5_checksum:
        raise ValueError(f'md5 checksum of local file for {url} does not match description: expected: {md5_checksum} but got {actual_md5_checksum}. Downloaded file could have been modified / corrupted, clean cache and retry...')

    def _open_url_and_load_gzip_file(url, data_home, n_retries, delay, arff_params):
        gzip_file = _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
        with closing(gzip_file):
            return load_arff_from_gzip_file(gzip_file, **arff_params)
    arff_params: Dict = dict(parser=parser, output_type=output_type, openml_columns_info=openml_columns_info, feature_names_to_select=feature_names_to_select, target_names_to_select=target_names_to_select, shape=shape, read_csv_kwargs=read_csv_kwargs or {})
    try:
        X, y, frame, categories = _open_url_and_load_gzip_file(url, data_home, n_retries, delay, arff_params)
    except Exception as exc:
        if parser != 'pandas':
            raise
        from pandas.errors import ParserError
        if not isinstance(exc, ParserError):
            raise
        arff_params['read_csv_kwargs'].update(quotechar="'")
        X, y, frame, categories = _open_url_and_load_gzip_file(url, data_home, n_retries, delay, arff_params)
    return (X, y, frame, categories)