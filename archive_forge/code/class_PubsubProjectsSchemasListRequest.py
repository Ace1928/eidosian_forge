from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSchemasListRequest(_messages.Message):
    """A PubsubProjectsSchemasListRequest object.

  Enums:
    ViewValueValuesEnum: The set of Schema fields to return in the response.
      If not set, returns Schemas with `name` and `type`, but not
      `definition`. Set to `FULL` to retrieve all fields.

  Fields:
    pageSize: Maximum number of schemas to return.
    pageToken: The value returned by the last `ListSchemasResponse`; indicates
      that this is a continuation of a prior `ListSchemas` call, and that the
      system should return the next page of data.
    parent: Required. The name of the project in which to list schemas. Format
      is `projects/{project-id}`.
    view: The set of Schema fields to return in the response. If not set,
      returns Schemas with `name` and `type`, but not `definition`. Set to
      `FULL` to retrieve all fields.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The set of Schema fields to return in the response. If not set,
    returns Schemas with `name` and `type`, but not `definition`. Set to
    `FULL` to retrieve all fields.

    Values:
      SCHEMA_VIEW_UNSPECIFIED: The default / unset value. The API will default
        to the BASIC view.
      BASIC: Include the name and type of the schema, but not the definition.
      FULL: Include all Schema object fields.
    """
        SCHEMA_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)