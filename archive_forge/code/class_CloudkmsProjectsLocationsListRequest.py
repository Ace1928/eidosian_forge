from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsListRequest(_messages.Message):
    """A CloudkmsProjectsLocationsListRequest object.

  Fields:
    filter: The standard list filter.
    name: The resource that owns the locations collection, if applicable.
    pageSize: The standard list page size.
    pageToken: The standard list page token.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)