from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def _make_regression_options(trading, exog):
    if not trading and exog is None:
        return ''
    reg_spec = 'regression{\n'
    if trading:
        reg_spec += '    variables = (td)\n'
    if exog is not None:
        var_names = _make_var_names(exog)
        reg_spec += f'    user = ({var_names})\n'
        reg_spec += '    data = ({})\n'.format('\n'.join(map(str, exog.values.ravel().tolist())))
    reg_spec += '}\n'
    return reg_spec