from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SnapshotSettingsAccessLocationAccessLocationPreference(_messages.Message):
    """A structure for specifying an allowed target region.

  Fields:
    region: Accessible region name
  """
    region = _messages.StringField(1)