import numpy as np
from numpy.testing import assert_equal
from statsmodels.iolib.table import Cell, SimpleTable
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
class TestCell:

    def test_celldata(self):
        celldata = (cell0data, cell1data, row1data[0], row1data[1])
        cells = [Cell(datum, datatype=i % 2) for i, datum in enumerate(celldata)]
        for cell, datum in zip(cells, celldata):
            assert_equal(cell.data, datum)