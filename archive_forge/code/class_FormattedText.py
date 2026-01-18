from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Tuple, Union, cast
from prompt_toolkit.mouse_events import MouseEvent
class FormattedText(StyleAndTextTuples):
    """
    A list of ``(style, text)`` tuples.

    (In some situations, this can also be ``(style, text, mouse_handler)``
    tuples.)
    """

    def __pt_formatted_text__(self) -> StyleAndTextTuples:
        return self

    def __repr__(self) -> str:
        return 'FormattedText(%s)' % super().__repr__()