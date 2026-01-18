from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsUrlListsDeleteRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsUrlListsDeleteRequest object.

  Fields:
    name: Required. A name of the UrlList to delete. Must be in the format
      `projects/*/locations/{location}/urlLists/*`.
  """
    name = _messages.StringField(1, required=True)