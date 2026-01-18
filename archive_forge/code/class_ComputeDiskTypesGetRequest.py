from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeDiskTypesGetRequest(_messages.Message):
    """A ComputeDiskTypesGetRequest object.

  Fields:
    diskType: Name of the disk type to return.
    project: Project ID for this request.
    zone: The name of the zone for this request.
  """
    diskType = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)