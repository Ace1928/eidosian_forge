import logging
import os
import shutil
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from mlflow.exceptions import BAD_REQUEST, INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.cards import pandas_renderer
from mlflow.utils.databricks_utils import (
from mlflow.utils.os import is_windows
def _get_pool_size():
    return 1 if 'PYTEST_CURRENT_TEST' in os.environ and is_windows() else 0