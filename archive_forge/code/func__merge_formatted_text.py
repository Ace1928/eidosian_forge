from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Tuple, Union, cast
from prompt_toolkit.mouse_events import MouseEvent
def _merge_formatted_text() -> AnyFormattedText:
    result = FormattedText()
    for i in items:
        result.extend(to_formatted_text(i))
    return result