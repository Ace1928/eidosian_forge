from pathlib import Path
from typing import Optional, List
import click
import logging
import operator
import os
import shutil
import subprocess
from datetime import datetime
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
from ray.air.constants import EXPR_RESULT_FILE
from ray.tune.result import (
from ray.tune.analysis import ExperimentAnalysis
from ray.tune import TuneError
from ray._private.thirdparty.tabulate.tabulate import tabulate
Opens a txt file at the given path where user can add and save notes.

    Args:
        path: Directory where note will be saved.
        filename: Name of note. Defaults to "note.txt"
    