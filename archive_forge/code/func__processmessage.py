from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Sequence
from typing import Tuple
def _processmessage(self, tags: tuple[str, ...], args: tuple[object, ...]) -> None:
    if self._writer is not None and args:
        self._writer(self._format_message(tags, args))
    try:
        processor = self._tags2proc[tags]
    except KeyError:
        pass
    else:
        processor(tags, args)