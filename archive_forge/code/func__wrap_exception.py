from typing import Dict, Tuple, Any
import traceback as tb
def _wrap_exception(exc: BaseException) -> WRAPPED_EXCEPTION:
    return (exc, tb.extract_tb(exc.__traceback__))