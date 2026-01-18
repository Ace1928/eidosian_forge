from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeUrlMapsValidateRequest(_messages.Message):
    """A ComputeUrlMapsValidateRequest object.

  Fields:
    project: Project ID for this request.
    urlMap: Name of the UrlMap resource to be validated as.
    urlMapsValidateRequest: A UrlMapsValidateRequest resource to be passed as
      the request body.
  """
    project = _messages.StringField(1, required=True)
    urlMap = _messages.StringField(2, required=True)
    urlMapsValidateRequest = _messages.MessageField('UrlMapsValidateRequest', 3)