from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
def FormatHrule(self):
    """Return a list containing an hrule for this table."""
    entries = (''.join(itertools.repeat('-', width + 2)) for width in self.column_widths)
    return [self.junction_char.join(itertools.chain([''], entries, ['']))]