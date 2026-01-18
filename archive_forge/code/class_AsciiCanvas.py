import math
import os
from typing import Any, Mapping, Sequence
from langchain_core.runnables.graph import Edge as LangEdge
class AsciiCanvas:
    """Class for drawing in ASCII.

    Args:
        cols (int): number of columns in the canvas. Should be > 1.
        lines (int): number of lines in the canvas. Should be > 1.
    """
    TIMEOUT = 10

    def __init__(self, cols: int, lines: int) -> None:
        assert cols > 1
        assert lines > 1
        self.cols = cols
        self.lines = lines
        self.canvas = [[' '] * cols for line in range(lines)]

    def draw(self) -> str:
        """Draws ASCII canvas on the screen."""
        lines = map(''.join, self.canvas)
        return os.linesep.join(lines)

    def point(self, x: int, y: int, char: str) -> None:
        """Create a point on ASCII canvas.

        Args:
            x (int): x coordinate. Should be >= 0 and < number of columns in
                the canvas.
            y (int): y coordinate. Should be >= 0 an < number of lines in the
                canvas.
            char (str): character to place in the specified point on the
                canvas.
        """
        assert len(char) == 1
        assert x >= 0
        assert x < self.cols
        assert y >= 0
        assert y < self.lines
        self.canvas[y][x] = char

    def line(self, x0: int, y0: int, x1: int, y1: int, char: str) -> None:
        """Create a line on ASCII canvas.

        Args:
            x0 (int): x coordinate where the line should start.
            y0 (int): y coordinate where the line should start.
            x1 (int): x coordinate where the line should end.
            y1 (int): y coordinate where the line should end.
            char (str): character to draw the line with.
        """
        if x0 > x1:
            x1, x0 = (x0, x1)
            y1, y0 = (y0, y1)
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0 and dy == 0:
            self.point(x0, y0, char)
        elif abs(dx) >= abs(dy):
            for x in range(x0, x1 + 1):
                if dx == 0:
                    y = y0
                else:
                    y = y0 + int(round((x - x0) * dy / float(dx)))
                self.point(x, y, char)
        elif y0 < y1:
            for y in range(y0, y1 + 1):
                if dy == 0:
                    x = x0
                else:
                    x = x0 + int(round((y - y0) * dx / float(dy)))
                self.point(x, y, char)
        else:
            for y in range(y1, y0 + 1):
                if dy == 0:
                    x = x0
                else:
                    x = x1 + int(round((y - y1) * dx / float(dy)))
                self.point(x, y, char)

    def text(self, x: int, y: int, text: str) -> None:
        """Print a text on ASCII canvas.

        Args:
            x (int): x coordinate where the text should start.
            y (int): y coordinate where the text should start.
            text (str): string that should be printed.
        """
        for i, char in enumerate(text):
            self.point(x + i, y, char)

    def box(self, x0: int, y0: int, width: int, height: int) -> None:
        """Create a box on ASCII canvas.

        Args:
            x0 (int): x coordinate of the box corner.
            y0 (int): y coordinate of the box corner.
            width (int): box width.
            height (int): box height.
        """
        assert width > 1
        assert height > 1
        width -= 1
        height -= 1
        for x in range(x0, x0 + width):
            self.point(x, y0, '-')
            self.point(x, y0 + height, '-')
        for y in range(y0, y0 + height):
            self.point(x0, y, '|')
            self.point(x0 + width, y, '|')
        self.point(x0, y0, '+')
        self.point(x0 + width, y0, '+')
        self.point(x0, y0 + height, '+')
        self.point(x0 + width, y0 + height, '+')