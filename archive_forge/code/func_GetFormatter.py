from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
def GetFormatter(table_format):
    """Map a format name to a TableFormatter object."""
    if table_format == 'csv':
        table_formatter = CsvFormatter()
    elif table_format == 'pretty':
        table_formatter = PrettyFormatter()
    elif table_format == 'json':
        table_formatter = JsonFormatter()
    elif table_format == 'prettyjson':
        table_formatter = PrettyJsonFormatter()
    elif table_format == 'sparse':
        table_formatter = SparsePrettyFormatter()
    elif table_format == 'none':
        table_formatter = NullFormatter()
    else:
        raise FormatterException('Unknown format: %s' % table_format)
    return table_formatter