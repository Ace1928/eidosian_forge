import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
class CannedCell(CannedObject):
    """Can a closure cell"""

    def __init__(self, cell):
        """Initialize the canned cell."""
        self.cell_contents = can(cell.cell_contents)

    def get_object(self, g=None):
        """Get an object in the cell."""
        cell_contents = uncan(self.cell_contents, g)

        def inner():
            """Inner function."""
            return cell_contents
        return inner.__closure__[0]