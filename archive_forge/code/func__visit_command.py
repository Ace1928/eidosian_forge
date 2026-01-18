import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _visit_command(self, item, x, y):
    """
        Return the bottom-rightmost point without actually drawing the item

        :param item: the item to visit
        :param x: the top of the current drawing area
        :param y: the left side of the current drawing area
        :return: the bottom-rightmost point
        """
    if isinstance(item, str):
        return (x + self.canvas.font.measure(item), y + self._get_text_height())
    elif isinstance(item, tuple):
        return item