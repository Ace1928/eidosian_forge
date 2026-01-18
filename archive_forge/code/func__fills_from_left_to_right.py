from kivy.logger import Logger
from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, BooleanProperty, DictProperty, \
from math import ceil
from itertools import accumulate, product, chain, islice
from operator import sub
@property
def _fills_from_left_to_right(self):
    return 'lr' in self.orientation