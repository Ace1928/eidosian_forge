import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Optional
import boto3
import numpy as np
import pandas
import pytest
import requests
import s3fs
from pandas.util._decorators import doc
import modin.utils  # noqa: E402
import uuid  # noqa: E402
import modin  # noqa: E402
import modin.config  # noqa: E402
import modin.tests.config  # noqa: E402
from modin.config import (  # noqa: E402
from modin.core.execution.dispatching.factories import factories  # noqa: E402
from modin.core.execution.python.implementations.pandas_on_python.io import (  # noqa: E402
from modin.core.storage_formats import (  # noqa: E402
from modin.tests.pandas.utils import (  # noqa: E402
def _sql_connection(filename, table=''):
    if os.path.exists(filename):
        os.remove(filename)
    conn = 'sqlite:///{}'.format(filename)
    if table:
        df = pandas.DataFrame({'col1': [0, 1, 2, 3, 4, 5, 6], 'col2': [7, 8, 9, 10, 11, 12, 13], 'col3': [14, 15, 16, 17, 18, 19, 20], 'col4': [21, 22, 23, 24, 25, 26, 27], 'col5': [0, 0, 0, 0, 0, 0, 0]})
        df.to_sql(table, conn)
    return conn