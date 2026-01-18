from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputePublicAdvertisedPrefixesGetRequest(_messages.Message):
    """A ComputePublicAdvertisedPrefixesGetRequest object.

  Fields:
    project: Project ID for this request.
    publicAdvertisedPrefix: Name of the PublicAdvertisedPrefix resource to
      return.
  """
    project = _messages.StringField(1, required=True)
    publicAdvertisedPrefix = _messages.StringField(2, required=True)