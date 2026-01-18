from typing import Any, List, Optional, Tuple
from fugue import FugueWorkflow
from fugue.exceptions import FugueDataFrameError
from triad import assert_or_throw
from tune._utils import from_base64
from tune.api.factory import TUNE_OBJECT_FACTORY, parse_logger
from tune.api.optimize import (
from tune.concepts.flow import TrialReport
from tune.concepts.logger import make_logger
from tune.concepts.space import Space
from tune.constants import TUNE_DATASET_DF_DEFAULT_NAME, TUNE_REPORT, TUNE_REPORT_METRIC
from tune.exceptions import TuneCompileError
def _safe(xx):
    return float('inf') if xx is None else float(xx)