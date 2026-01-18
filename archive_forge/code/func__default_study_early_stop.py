from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from triad import SerializableRLock, to_uuid
from tune.concepts.flow import (
def _default_study_early_stop(keys: List[Any], rungs: List['RungHeap']) -> bool:
    return all((r.full for r in rungs))