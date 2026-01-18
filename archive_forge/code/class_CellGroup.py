import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
class CellGroup:

    def __init__(self, cells):
        self.cells = cells
        self.bbox = (min(map(itemgetter(0), filter(None, cells))), min(map(itemgetter(1), filter(None, cells))), max(map(itemgetter(2), filter(None, cells))), max(map(itemgetter(3), filter(None, cells))))