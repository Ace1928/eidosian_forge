from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from utils import bq_error
def ReadRows(self, start_row: Optional[int]=0, max_rows: Optional[int]=None, selected_fields=None):
    """Read at most max_rows rows from a table.

    Args:
      start_row: first row to return.
      max_rows: maximum number of rows to return.
      selected_fields: a subset of fields to return.

    Raises:
      BigqueryInterfaceError: when bigquery returns something unexpected.

    Returns:
      list of rows, each of which is a list of field values.
    """
    _, rows = self.ReadSchemaAndRows(start_row=start_row, max_rows=max_rows, selected_fields=selected_fields)
    return rows