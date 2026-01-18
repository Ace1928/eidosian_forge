from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionUrlMapsValidateRequest(_messages.Message):
    """A ComputeRegionUrlMapsValidateRequest object.

  Fields:
    project: Project ID for this request.
    region: Name of the region scoping this request.
    regionUrlMapsValidateRequest: A RegionUrlMapsValidateRequest resource to
      be passed as the request body.
    urlMap: Name of the UrlMap resource to be validated as.
  """
    project = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    regionUrlMapsValidateRequest = _messages.MessageField('RegionUrlMapsValidateRequest', 3)
    urlMap = _messages.StringField(4, required=True)