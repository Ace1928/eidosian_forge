from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
class InlineElement(ABC):
    """Arbitrary inline element positioned within a formatted document.

    Elements behave like a single glyph in the document.  They are
    measured by their horizontal advance, ascent above the baseline, and
    descent below the baseline.

    The pyglet layout classes reserve space in the layout for elements and
    call the element's methods to ensure they are rendered at the
    appropriate position.

    If the size of a element (any of the `advance`, `ascent`, or `descent`
    instance variables) is modified it is the application's responsibility to
    trigger a reflow of the appropriate area in the affected layouts.  This
    can be done by forcing a style change over the element's position.

    :Ivariables:
        `ascent` : int
            Ascent of the element above the baseline, in pixels.
        `descent` : int
            Descent of the element below the baseline, in pixels.
            Typically negative.
        `advance` : int
            Width of the element, in pixels.

    """

    def __init__(self, ascent: int, descent: int, advance: int) -> None:
        self.ascent = ascent
        self.descent = descent
        self.advance = advance
        self._position = None

    @property
    def position(self) -> Optional[int]:
        return self._position

    @abstractmethod
    def place(self, layout: TextLayout, x: float, y: float, z: float, line_x: float, line_y: float, rotation: float, visible: bool, anchor_x: float, anchor_y: float) -> None:
        ...

    @abstractmethod
    def update_translation(self, x: float, y: float, z: float) -> None:
        ...

    @abstractmethod
    def update_color(self, color: List[int]) -> None:
        ...

    @abstractmethod
    def update_view_translation(self, translate_x: float, translate_y: float) -> None:
        ...

    @abstractmethod
    def update_rotation(self, rotation: float) -> None:
        ...

    @abstractmethod
    def update_visibility(self, visible: bool) -> None:
        ...

    @abstractmethod
    def update_anchor(self, anchor_x: float, anchor_y: float) -> None:
        ...

    @abstractmethod
    def remove(self, layout: TextLayout) -> None:
        ...