from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
class TableSchemaFormatter(BaseFormatter):
    print_method = ObjectName('_repr_data_resource_')
    _return_type = (dict,)