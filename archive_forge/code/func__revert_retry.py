import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
def _revert_retry(retry, arguments):
    try:
        result = retry.revert(**arguments)
    except Exception:
        result = failure.Failure()
    return (REVERTED, result)