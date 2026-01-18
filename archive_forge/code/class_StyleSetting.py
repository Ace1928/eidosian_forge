from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class StyleSetting(_messages.Message):
    """Represents a complete StyleSettings object. The primary key is a
  combination of the tableId and a styleId.

  Fields:
    kind: Type name: an individual style setting. A StyleSetting contains the
      style defintions for points, lines, and polygons in a table. Since a
      table can have any one or all of them, a style definition can have
      point, line and polygon style definitions.
    markerOptions: Style definition for points in the table.
    name: Optional name for the style setting.
    polygonOptions: Style definition for polygons in the table.
    polylineOptions: Style definition for lines in the table.
    styleId: Identifier for the style setting (unique only within tables).
    tableId: Identifier for the table.
  """
    kind = _messages.StringField(1, default=u'fusiontables#styleSetting')
    markerOptions = _messages.MessageField('PointStyle', 2)
    name = _messages.StringField(3)
    polygonOptions = _messages.MessageField('PolygonStyle', 4)
    polylineOptions = _messages.MessageField('LineStyle', 5)
    styleId = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    tableId = _messages.StringField(7)