from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageObjectsListRequest(_messages.Message):
    """A StorageObjectsListRequest object.

  Enums:
    ProjectionValueValuesEnum: Set of properties to return. Defaults to noAcl.

  Fields:
    bucket: Name of the bucket in which to look for objects.
    delimiter: Returns results in a directory-like mode. items will contain
      only objects whose names, aside from the prefix, do not contain
      delimiter. Objects whose names, aside from the prefix, contain delimiter
      will have their name, truncated after the delimiter, returned in
      prefixes. Duplicate prefixes are omitted.
    includeTrailingDelimiter: If true, objects that end in exactly one
      instance of delimiter will have their metadata included in items in
      addition to prefixes.
    maxResults: Maximum number of items plus prefixes to return in a single
      page of responses. As duplicate prefixes are omitted, fewer total
      results may be returned than requested. The service will use this
      parameter or 1,000 items, whichever is smaller.
    pageToken: A previously-returned page token representing part of the
      larger set of results to view.
    prefix: Filter results to objects whose names begin with this prefix.
    projection: Set of properties to return. Defaults to noAcl.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
    versions: If true, lists all versions of an object as distinct results.
      The default is false. For more information, see Object Versioning.
  """

    class ProjectionValueValuesEnum(_messages.Enum):
        """Set of properties to return. Defaults to noAcl.

    Values:
      full: Include all properties.
      noAcl: Omit the owner, acl property.
    """
        full = 0
        noAcl = 1
    bucket = _messages.StringField(1, required=True)
    delimiter = _messages.StringField(2)
    includeTrailingDelimiter = _messages.BooleanField(3)
    maxResults = _messages.IntegerField(4, variant=_messages.Variant.UINT32, default=1000)
    pageToken = _messages.StringField(5)
    prefix = _messages.StringField(6)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 7)
    userProject = _messages.StringField(8)
    versions = _messages.BooleanField(9)