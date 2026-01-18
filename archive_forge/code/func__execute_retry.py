import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
def _execute_retry(retry, arguments):
    try:
        result = retry.execute(**arguments)
    except Exception:
        result = failure.Failure()
    return (EXECUTED, result)