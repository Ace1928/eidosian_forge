from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageinsightsProjectsLocationsDatasetConfigsLinkDatasetRequest(_messages.Message):
    """A StorageinsightsProjectsLocationsDatasetConfigsLinkDatasetRequest
  object.

  Fields:
    linkDatasetRequest: A LinkDatasetRequest resource to be passed as the
      request body.
    name: Required. Name of the resource
  """
    linkDatasetRequest = _messages.MessageField('LinkDatasetRequest', 1)
    name = _messages.StringField(2, required=True)