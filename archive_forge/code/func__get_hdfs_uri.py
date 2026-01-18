import os
import random
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _test_dataframe
from pyarrow.tests.parquet.test_dataset import (
from pyarrow.util import guid
def _get_hdfs_uri(path):
    host = os.environ.get('ARROW_HDFS_TEST_HOST', 'localhost')
    try:
        port = int(os.environ.get('ARROW_HDFS_TEST_PORT', 0))
    except ValueError:
        raise ValueError('Env variable ARROW_HDFS_TEST_PORT was not an integer')
    uri = 'hdfs://{}:{}{}'.format(host, port, path)
    return uri