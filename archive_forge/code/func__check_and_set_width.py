from .links_base import Strand, Crossing, Link
import random
import collections
def _check_and_set_width(self):
    width = 0
    max_width = 0
    for event in self.events:
        if event.kind == 'cup':
            assert event.max < width + 2
            width += 2
            max_width = max(width, max_width)
        elif event.kind == 'cap':
            assert event.max < width
            width += -2
        elif event.kind == 'cross':
            assert event.max < width
    assert width == 0
    self.width = max_width