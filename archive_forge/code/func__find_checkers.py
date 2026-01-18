from __future__ import annotations
from collections.abc import Callable
from babel.messages.catalog import PYTHON_FORMAT, Catalog, Message, TranslationError
def _find_checkers() -> list[Callable[[Catalog | None, Message], object]]:
    checkers: list[Callable[[Catalog | None, Message], object]] = []
    try:
        from pkg_resources import working_set
    except ImportError:
        pass
    else:
        for entry_point in working_set.iter_entry_points('babel.checkers'):
            checkers.append(entry_point.load())
    if len(checkers) == 0:
        return [num_plurals, python_format]
    return checkers