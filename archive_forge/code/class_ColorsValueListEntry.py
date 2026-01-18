from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class ColorsValueListEntry(_messages.Message):
    """A ColorsValueListEntry object.

      Fields:
        color: Color in #RRGGBB format.
        opacity: Opacity of the color: 0.0 (transparent) to 1.0 (opaque).
      """
    color = _messages.StringField(1)
    opacity = _messages.FloatField(2)