from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MosaicLayout(_messages.Message):
    """A mosaic layout divides the available space into a grid of blocks, and
  overlays the grid with tiles. Unlike GridLayout, tiles may span multiple
  grid blocks and can be placed at arbitrary locations in the grid.

  Fields:
    columns: The number of columns in the mosaic grid. The number of columns
      must be between 1 and 12, inclusive.
    tiles: The tiles to display.
  """
    columns = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    tiles = _messages.MessageField('Tile', 2, repeated=True)