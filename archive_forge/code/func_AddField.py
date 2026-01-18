from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
def AddField(self, field):
    """Add a field as a new column to this formatter."""
    align = 'l' if field.get('type', []) == 'STRING' else 'r'
    self.AddColumn(field['name'], align=align)