from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def _check_errors(errors):
    errors = errors[errors.find('spc:') + 4:].strip()
    if errors and 'ERROR' in errors:
        raise X13Error(errors)
    elif errors and 'WARNING' in errors:
        warn(errors, X13Warning)