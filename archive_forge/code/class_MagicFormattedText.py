from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Tuple, Union, cast
from prompt_toolkit.mouse_events import MouseEvent
class MagicFormattedText(Protocol):
    """
        Any object that implements ``__pt_formatted_text__`` represents formatted
        text.
        """

    def __pt_formatted_text__(self) -> StyleAndTextTuples:
        ...