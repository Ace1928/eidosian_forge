from kivy.logger import Logger
from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, BooleanProperty, DictProperty, \
from math import ceil
from itertools import accumulate, product, chain, islice
from operator import sub
class GridLayoutException(Exception):
    """Exception for errors if the grid layout manipulation fails.
    """
    pass