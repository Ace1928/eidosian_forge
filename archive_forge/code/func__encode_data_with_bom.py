from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def _encode_data_with_bom(_data):
    bom_data = (bom + _data).encode(utf8)
    return BytesIO(bom_data)