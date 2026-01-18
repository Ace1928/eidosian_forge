from __future__ import annotations
from collections.abc import Iterable
from typing import Final, TYPE_CHECKING, Callable
import numpy as np
def _override_imports(file: MypyFile, module: str, imports: list[tuple[str, None | str]]) -> None:
    """Override the first `module`-based import with new `imports`."""
    import_obj = ImportFrom(module, 0, names=imports)
    import_obj.is_top_level = True
    for lst in [file.defs, file.imports]:
        i = _index(lst, module)
        lst[i] = import_obj