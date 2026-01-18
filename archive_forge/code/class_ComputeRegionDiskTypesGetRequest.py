from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionDiskTypesGetRequest(_messages.Message):
    """A ComputeRegionDiskTypesGetRequest object.

  Fields:
    diskType: Name of the disk type to return.
    project: Project ID for this request.
    region: The name of the region for this request.
  """
    diskType = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)