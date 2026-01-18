from itertools import groupby
import numpy as np
import pandas as pd
import param
from .dimension import Dimensioned, ViewableElement, asdim
from .layout import Composable, Layout, NdLayout
from .ndmapping import NdMapping
from .overlay import CompositeOverlay, NdOverlay, Overlayable
from .spaces import GridSpace, HoloMap
from .tree import AttrTree
from .util import get_param_values
class Tabular(Element):
    """
    Baseclass to give an elements providing an API to generate a
    tabular representation of the object.
    """
    __abstract = True

    @property
    def rows(self):
        """Number of rows in table (including header)"""
        return len(self) + 1

    @property
    def cols(self):
        """Number of columns in table"""
        return len(self.dimensions())

    def pprint_cell(self, row, col):
        """Formatted contents of table cell.

        Args:
            row (int): Integer index of table row
            col (int): Integer index of table column

        Returns:
            Formatted table cell contents
        """
        ndims = self.ndims
        if col >= self.cols:
            raise Exception('Maximum column index is %d' % self.cols - 1)
        elif row >= self.rows:
            raise Exception('Maximum row index is %d' % self.rows - 1)
        elif row == 0:
            if col >= ndims:
                if self.vdims:
                    return self.vdims[col - ndims].pprint_label
                else:
                    return ''
            return self.kdims[col].pprint_label
        else:
            dim = self.get_dimension(col)
            return dim.pprint_value(self.iloc[row - 1, col])

    def cell_type(self, row, col):
        """Type of the table cell, either 'data' or 'heading'

        Args:
            row (int): Integer index of table row
            col (int): Integer index of table column

        Returns:
            Type of the table cell, either 'data' or 'heading'
        """
        return 'heading' if row == 0 else 'data'