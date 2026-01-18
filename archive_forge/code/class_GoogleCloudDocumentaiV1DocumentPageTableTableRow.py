from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentPageTableTableRow(_messages.Message):
    """A row of table cells.

  Fields:
    cells: Cells that make up this row.
  """
    cells = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageTableTableCell', 1, repeated=True)