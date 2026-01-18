from __future__ import annotations
from collections.abc import Callable
from babel.messages.catalog import PYTHON_FORMAT, Catalog, Message, TranslationError
def _check_positional(results: list[tuple[str, str]]) -> bool:
    positional = None
    for name, _char in results:
        if positional is None:
            positional = name is None
        elif (name is None) != positional:
            raise TranslationError('format string mixes positional and named placeholders')
    return bool(positional)