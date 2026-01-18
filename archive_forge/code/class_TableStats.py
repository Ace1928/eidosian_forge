from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableStats(_messages.Message):
    """Approximate statistics related to a table. These statistics are
  calculated infrequently, while simultaneously, data in the table can change
  rapidly. Thus the values reported here (e.g. row count) are very likely out-
  of date, even the instant they are received in this API. Thus, only treat
  these values as approximate. IMPORTANT: Everything below is approximate,
  unless otherwise specified.

  Fields:
    averageCellsPerColumn: How many cells are present per column (column
      family, column qualifier) combinations, averaged over all columns in all
      rows in the table. e.g. A table with 2 rows: * A row with 3 cells in
      "family:col" and 1 cell in "other:col" (4 cells / 2 columns) * A row
      with 1 cell in "family:col", 7 cells in "family:other_col", and 7 cells
      in "other:data" (15 cells / 3 columns) would report (4 + 15)/(2 + 3) =
      3.8 in this field.
    averageColumnsPerRow: How many (column family, column qualifier)
      combinations are present per row in the table, averaged over all rows in
      the table. e.g. A table with 2 rows: * A row with cells in "family:col"
      and "other:col" (2 distinct columns) * A row with cells in "family:col",
      "family:other_col", and "other:data" (3 distinct columns) would report
      (2 + 3)/2 = 2.5 in this field.
    logicalDataBytes: This is roughly how many bytes would be needed to read
      the entire table (e.g. by streaming all contents out).
    rowCount: How many rows are in the table.
  """
    averageCellsPerColumn = _messages.FloatField(1)
    averageColumnsPerRow = _messages.FloatField(2)
    logicalDataBytes = _messages.IntegerField(3)
    rowCount = _messages.IntegerField(4)