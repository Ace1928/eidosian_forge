from typing import Dict, Tuple, Any
import traceback as tb
def _is_wrapped_exception(obj: Any) -> bool:
    if not isinstance(obj, tuple):
        return False
    if len(obj) != 2:
        return False
    return isinstance(obj[0], BaseException) and isinstance(obj[1], tb.StackSummary)