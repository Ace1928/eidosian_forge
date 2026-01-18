from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionUrlMapsValidateRequest(_messages.Message):
    """A RegionUrlMapsValidateRequest object.

  Fields:
    resource: Content of the UrlMap to be validated.
  """
    resource = _messages.MessageField('UrlMap', 1)