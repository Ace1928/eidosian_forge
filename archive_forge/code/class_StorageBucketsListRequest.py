from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBucketsListRequest(_messages.Message):
    """A StorageBucketsListRequest object.

  Enums:
    ProjectionValueValuesEnum: Set of properties to return. Defaults to noAcl.

  Fields:
    maxResults: Maximum number of buckets to return in a single response. The
      service will use this parameter or 1,000 items, whichever is smaller.
    pageToken: A previously-returned page token representing part of the
      larger set of results to view.
    prefix: Filter results to buckets whose names begin with this prefix.
    project: A valid API project identifier.
    projection: Set of properties to return. Defaults to noAcl.
    userProject: The project to be billed for this request.
  """

    class ProjectionValueValuesEnum(_messages.Enum):
        """Set of properties to return. Defaults to noAcl.

    Values:
      full: Include all properties.
      noAcl: Omit owner, acl and defaultObjectAcl properties.
    """
        full = 0
        noAcl = 1
    maxResults = _messages.IntegerField(1, variant=_messages.Variant.UINT32, default=1000)
    pageToken = _messages.StringField(2)
    prefix = _messages.StringField(3)
    project = _messages.StringField(4, required=True)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 5)
    userProject = _messages.StringField(6)