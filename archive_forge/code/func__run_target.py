from multiprocessing import Pool
from typing import Any, Callable, Dict, List, TypeVar
import cloudpickle
from triad.utils.convert import to_timedelta
from tune.constants import TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
from tune.exceptions import TuneInterrupted
def _run_target(blob: Any):
    tp = cloudpickle.loads(blob)
    return cloudpickle.dumps(tp[0](*tp[1], **tp[2]))