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
def _open_cache(cache_path):
    import zlib
    with open(cache_path, 'rb') as zf:
        return zlib.decompress(zf.read())