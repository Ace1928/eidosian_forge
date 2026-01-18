from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _DisplayNumberOfRowsModified(row_count, is_exact_count, out):
    """Prints number of rows modified by a DML statement.

  Args:
    row_count: Either the exact number of rows modified by statement or the
      lower bound of rows modified by a Partitioned DML statement.
    is_exact_count: Boolean stating whether the number is the exact count.
    out: Output stream to which we print.
  """
    if is_exact_count:
        output_str = 'Statement modified {} {}'
    else:
        output_str = 'Statement modified a lower bound of {} {}'
    if row_count == 1:
        out.Print(output_str.format(row_count, 'row'))
    else:
        out.Print(output_str.format(row_count, 'rows'))