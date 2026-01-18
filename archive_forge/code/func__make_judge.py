from copy import copy
from typing import Any, Callable, Dict, Iterable, Optional
from fugue import ArrayDataFrame, DataFrame, ExecutionEngine
from triad import assert_or_throw
from tune._utils import run_monitored_process
from tune.concepts.dataset import StudyResult, TuneDataset, _get_trials_from_row
from tune.concepts.flow import RemoteTrialJudge, TrialCallback, TrialJudge, TrialReport
from tune.concepts.flow.judge import Monitor, NoOpTrailJudge
from tune.constants import TUNE_REPORT_ADD_SCHEMA, TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
from tune.exceptions import TuneCompileError, TuneInterrupted
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
def _make_judge(monitor: Optional[Monitor]=None, stopper: Optional[NonIterativeStopper]=None) -> Optional[TrialJudge]:
    if monitor is None and stopper is None:
        return None
    if stopper is None and monitor is not None:
        return NoOpTrailJudge(monitor)
    if stopper is not None and monitor is None:
        return stopper
    if stopper is not None and monitor is not None:
        stopper.reset_monitor(monitor)
        return stopper
    raise NotImplementedError