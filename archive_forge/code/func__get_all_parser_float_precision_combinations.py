from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
def _get_all_parser_float_precision_combinations():
    """
    Return all allowable parser and float precision
    combinations and corresponding ids.
    """
    params = []
    ids = []
    for parser, parser_id in zip(_all_parsers, _all_parser_ids):
        if hasattr(parser, 'values'):
            parser = parser.values[0]
        for precision in parser.float_precision_choices:
            mark = pytest.mark.single_cpu if parser.engine == 'pyarrow' else ()
            param = pytest.param((parser(), precision), marks=mark)
            params.append(param)
            ids.append(f'{parser_id}-{precision}')
    return {'params': params, 'ids': ids}