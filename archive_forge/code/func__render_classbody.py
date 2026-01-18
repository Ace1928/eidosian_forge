from __future__ import annotations
from collections.abc import Iterator, Sequence
from typing import Optional
from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Any, Callable, overload
from xarray.core import nputils, ops
from xarray.core.types import (
def _render_classbody(method_blocks: list[OpsType]) -> Iterator[str]:
    for method_func_pairs, template, extra in method_blocks:
        if template:
            for method, func in method_func_pairs:
                yield template.format(method=method, func=func, **extra)
    yield ''
    for method_func_pairs, *_ in method_blocks:
        for method, func in method_func_pairs:
            if method and func:
                yield COPY_DOCSTRING.format(method=method, func=func)