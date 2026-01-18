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
def _get_data_info_by_name(name: str, version: Union[int, str], data_home: Optional[str], n_retries: int=3, delay: float=1.0):
    """
    Utilizes the openml dataset listing api to find a dataset by
    name/version
    OpenML api function:
    https://www.openml.org/api_docs#!/data/get_data_list_data_name_data_name

    Parameters
    ----------
    name : str
        name of the dataset

    version : int or str
        If version is an integer, the exact name/version will be obtained from
        OpenML. If version is a string (value: "active") it will take the first
        version from OpenML that is annotated as active. Any other string
        values except "active" are treated as integer.

    data_home : str or None
        Location to cache the response. None if no cache is required.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    first_dataset : json
        json representation of the first dataset object that adhired to the
        search criteria

    """
    if version == 'active':
        url = _SEARCH_NAME.format(name) + '/status/active/'
        error_msg = 'No active dataset {} found.'.format(name)
        json_data = _get_json_content_from_openml_api(url, error_msg, data_home=data_home, n_retries=n_retries, delay=delay)
        res = json_data['data']['dataset']
        if len(res) > 1:
            first_version = version = res[0]['version']
            warning_msg = f'Multiple active versions of the dataset matching the name {name} exist. Versions may be fundamentally different, returning version {first_version}. Available versions:\n'
            for r in res:
                warning_msg += f'- version {r['version']}, status: {r['status']}\n'
                warning_msg += f'  url: https://www.openml.org/search?type=data&id={r['did']}\n'
            warn(warning_msg)
        return res[0]
    url = (_SEARCH_NAME + '/data_version/{}').format(name, version)
    try:
        json_data = _get_json_content_from_openml_api(url, error_message=None, data_home=data_home, n_retries=n_retries, delay=delay)
    except OpenMLError:
        url += '/status/deactivated'
        error_msg = 'Dataset {} with version {} not found.'.format(name, version)
        json_data = _get_json_content_from_openml_api(url, error_msg, data_home=data_home, n_retries=n_retries, delay=delay)
    return json_data['data']['dataset'][0]