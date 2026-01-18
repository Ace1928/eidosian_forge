import gzip
import json
import os
import re
from functools import partial
from importlib import resources
from io import BytesIO
from urllib.error import HTTPError
import numpy as np
import pytest
import scipy.sparse
import sklearn
from sklearn import config_context
from sklearn.datasets import fetch_openml as fetch_openml_orig
from sklearn.datasets._openml import (
from sklearn.utils import Bunch, check_pandas_support
from sklearn.utils._testing import (
def _monkey_patch_webbased_functions(context, data_id, gzip_response):
    url_prefix_data_description = 'https://api.openml.org/api/v1/json/data/'
    url_prefix_data_features = 'https://api.openml.org/api/v1/json/data/features/'
    url_prefix_download_data = 'https://api.openml.org/data/v1/'
    url_prefix_data_list = 'https://api.openml.org/api/v1/json/data/list/'
    path_suffix = '.gz'
    read_fn = gzip.open
    data_module = OPENML_TEST_DATA_MODULE + '.' + f'id_{data_id}'

    def _file_name(url, suffix):
        output = re.sub('\\W', '-', url[len('https://api.openml.org/'):]) + suffix + path_suffix
        return output.replace('-json-data-list', '-jdl').replace('-json-data-features', '-jdf').replace('-json-data-qualities', '-jdq').replace('-json-data', '-jd').replace('-data_name', '-dn').replace('-download', '-dl').replace('-limit', '-l').replace('-data_version', '-dv').replace('-status', '-s').replace('-deactivated', '-dact').replace('-active', '-act')

    def _mock_urlopen_shared(url, has_gzip_header, expected_prefix, suffix):
        assert url.startswith(expected_prefix)
        data_file_name = _file_name(url, suffix)
        data_file_path = resources.files(data_module) / data_file_name
        with data_file_path.open('rb') as f:
            if has_gzip_header and gzip_response:
                fp = BytesIO(f.read())
                return _MockHTTPResponse(fp, True)
            else:
                decompressed_f = read_fn(f, 'rb')
                fp = BytesIO(decompressed_f.read())
                return _MockHTTPResponse(fp, False)

    def _mock_urlopen_data_description(url, has_gzip_header):
        return _mock_urlopen_shared(url=url, has_gzip_header=has_gzip_header, expected_prefix=url_prefix_data_description, suffix='.json')

    def _mock_urlopen_data_features(url, has_gzip_header):
        return _mock_urlopen_shared(url=url, has_gzip_header=has_gzip_header, expected_prefix=url_prefix_data_features, suffix='.json')

    def _mock_urlopen_download_data(url, has_gzip_header):
        return _mock_urlopen_shared(url=url, has_gzip_header=has_gzip_header, expected_prefix=url_prefix_download_data, suffix='.arff')

    def _mock_urlopen_data_list(url, has_gzip_header):
        assert url.startswith(url_prefix_data_list)
        data_file_name = _file_name(url, '.json')
        data_file_path = resources.files(data_module) / data_file_name
        with data_file_path.open('rb') as f:
            decompressed_f = read_fn(f, 'rb')
            decoded_s = decompressed_f.read().decode('utf-8')
            json_data = json.loads(decoded_s)
        if 'error' in json_data:
            raise HTTPError(url=None, code=412, msg='Simulated mock error', hdrs=None, fp=BytesIO())
        with data_file_path.open('rb') as f:
            if has_gzip_header:
                fp = BytesIO(f.read())
                return _MockHTTPResponse(fp, True)
            else:
                decompressed_f = read_fn(f, 'rb')
                fp = BytesIO(decompressed_f.read())
                return _MockHTTPResponse(fp, False)

    def _mock_urlopen(request, *args, **kwargs):
        url = request.get_full_url()
        has_gzip_header = request.get_header('Accept-encoding') == 'gzip'
        if url.startswith(url_prefix_data_list):
            return _mock_urlopen_data_list(url, has_gzip_header)
        elif url.startswith(url_prefix_data_features):
            return _mock_urlopen_data_features(url, has_gzip_header)
        elif url.startswith(url_prefix_download_data):
            return _mock_urlopen_download_data(url, has_gzip_header)
        elif url.startswith(url_prefix_data_description):
            return _mock_urlopen_data_description(url, has_gzip_header)
        else:
            raise ValueError('Unknown mocking URL pattern: %s' % url)
    if test_offline:
        context.setattr(sklearn.datasets._openml, 'urlopen', _mock_urlopen)