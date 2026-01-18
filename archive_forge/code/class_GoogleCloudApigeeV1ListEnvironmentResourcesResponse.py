from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListEnvironmentResourcesResponse(_messages.Message):
    """Response for ListEnvironmentResources

  Fields:
    resourceFile: List of resources files.
  """
    resourceFile = _messages.MessageField('GoogleCloudApigeeV1ResourceFile', 1, repeated=True)