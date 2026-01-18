from io import (
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
class NoSeekTellBuffer(StringIO):

    def tell(self):
        raise AttributeError('No tell method')

    def seek(self, pos, whence=0):
        raise AttributeError('No seek method')