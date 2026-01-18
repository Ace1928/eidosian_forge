import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _draw_command(self, item, x, y):
    """
        Draw the given item at the given location

        :param item: the item to draw
        :param x: the top of the current drawing area
        :param y: the left side of the current drawing area
        :return: the bottom-rightmost point
        """
    if isinstance(item, str):
        self.canvas.create_text(x, y, anchor='nw', font=self.canvas.font, text=item)
    elif isinstance(item, tuple):
        right, bottom = item
        self.canvas.create_rectangle(x, y, right, bottom)
        horiz_line_y = y + self._get_text_height() + self.BUFFER * 2
        self.canvas.create_line(x, horiz_line_y, right, horiz_line_y)
    return self._visit_command(item, x, y)