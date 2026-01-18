from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
def AddDict(self, d):
    """Add a dict as a row by using column names as keys."""
    self.AddRow([d.get(name, '') for name in self.column_names])