from statsmodels.compat.python import lrange
from io import StringIO
from os import environ, makedirs
from os.path import abspath, dirname, exists, expanduser, join
import shutil
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen
import numpy as np
from pandas import Index, read_csv, read_stata
def _get_dataset_meta(dataname, package, cache):
    index_url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/datasets.csv'
    data, _ = _urlopen_cached(index_url, cache)
    data = data.decode('utf-8', 'strict')
    index = read_csv(StringIO(data))
    idx = np.logical_and(index.Item == dataname, index.Package == package)
    if not idx.any():
        raise ValueError(f'Item {dataname} from Package {package} was not found. Check the CSV file at {index_url} to verify the Item and Package.')
    dataset_meta = index.loc[idx]
    return dataset_meta['Title'].iloc[0]