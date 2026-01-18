from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
def FormatRows(self):
    """Return an iterator over all the rows in this table."""
    return itertools.chain(*map(self.FormatRow, self.rows, self.row_heights))