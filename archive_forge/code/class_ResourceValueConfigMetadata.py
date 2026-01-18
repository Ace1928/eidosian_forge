from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceValueConfigMetadata(_messages.Message):
    """Metadata about a ResourceValueConfig. For example, id and name.

  Fields:
    name: Resource value config name
  """
    name = _messages.StringField(1)