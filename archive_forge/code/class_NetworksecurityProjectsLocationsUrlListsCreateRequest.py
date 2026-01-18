from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsUrlListsCreateRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsUrlListsCreateRequest object.

  Fields:
    parent: Required. The parent resource of the UrlList. Must be in the
      format `projects/*/locations/{location}`.
    urlList: A UrlList resource to be passed as the request body.
    urlListId: Required. Short name of the UrlList resource to be created.
      This value should be 1-63 characters long, containing only letters,
      numbers, hyphens, and underscores, and should not start with a number.
      E.g. "url_list".
  """
    parent = _messages.StringField(1, required=True)
    urlList = _messages.MessageField('UrlList', 2)
    urlListId = _messages.StringField(3)