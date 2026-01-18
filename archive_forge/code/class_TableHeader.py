import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
class TableHeader:
    """PyMuPDF extension containing the identified table header."""

    def __init__(self, bbox, cells, names, above):
        self.bbox = bbox
        self.cells = cells
        self.names = names
        self.external = above