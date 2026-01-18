from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class PointStyle(_messages.Message):
    """Represents a PointStyle within a StyleSetting

  Fields:
    iconName: Name of the icon. Use values defined in
      http://www.google.com/fusiontables/DataSource?dsrcid=308519
    iconStyler: Column or a bucket value from which the icon name is to be
      determined.
  """
    iconName = _messages.StringField(1)
    iconStyler = _messages.MessageField('StyleFunction', 2)