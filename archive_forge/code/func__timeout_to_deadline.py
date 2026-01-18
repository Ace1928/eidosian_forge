import time
from typing import Optional
def _timeout_to_deadline(timeout: Optional[float]) -> Optional[float]:
    if timeout is None:
        return None
    return time.time() + timeout