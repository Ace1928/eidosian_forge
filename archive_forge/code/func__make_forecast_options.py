from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def _make_forecast_options(forecast_periods):
    if forecast_periods is None:
        return ''
    forecast_spec = 'forecast{\n'
    forecast_spec += f'maxlead = ({forecast_periods})\n}}\n'
    return forecast_spec