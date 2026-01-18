import os
import sys
from xml.dom.minidom import parse
from xml.dom.minidom import Node
class CmpTuple:
    """Compare function between 2 tuple."""

    def __call__(self, x, y):
        return cmp(x[0], y[0])