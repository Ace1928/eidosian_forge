from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductsPurgeRequest(_messages.Message):
    """A VisionProjectsLocationsProductsPurgeRequest object.

  Fields:
    parent: Required. The project and location in which the Products should be
      deleted. Format is `projects/PROJECT_ID/locations/LOC_ID`.
    purgeProductsRequest: A PurgeProductsRequest resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    purgeProductsRequest = _messages.MessageField('PurgeProductsRequest', 2)