import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def _get_open_files():
    """
    psutil open_files() can return a lot of extra information that we can allow to
    be different, like file position; for simplicity we care about path and fd only.
    """
    return sorted(((info.path, info.fd) for info in psutil.Process().open_files()))