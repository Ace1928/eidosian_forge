from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Lake(_messages.Message):
    """Represents a Lake resource

  Fields:
    name: The Lake resource name. Example:
      projects/{project_number}/locations/{location_id}/lakes/{lake_id}
  """
    name = _messages.StringField(1)