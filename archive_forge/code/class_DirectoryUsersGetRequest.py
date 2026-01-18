from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersGetRequest(_messages.Message):
    """A DirectoryUsersGetRequest object.

  Enums:
    ProjectionValueValuesEnum: What subset of fields to fetch for this user.
    ViewTypeValueValuesEnum: Whether to fetch the ADMIN_VIEW or DOMAIN_PUBLIC
      view of the user.

  Fields:
    customFieldMask: Comma-separated list of schema names. All fields from
      these schemas are fetched. This should only be set when
      projection=custom.
    projection: What subset of fields to fetch for this user.
    userKey: Email or immutable ID of the user
    viewType: Whether to fetch the ADMIN_VIEW or DOMAIN_PUBLIC view of the
      user.
  """

    class ProjectionValueValuesEnum(_messages.Enum):
        """What subset of fields to fetch for this user.

    Values:
      basic: Do not include any custom fields for the user.
      custom: Include custom fields from schemas mentioned in customFieldMask.
      full: Include all fields associated with this user.
    """
        basic = 0
        custom = 1
        full = 2

    class ViewTypeValueValuesEnum(_messages.Enum):
        """Whether to fetch the ADMIN_VIEW or DOMAIN_PUBLIC view of the user.

    Values:
      admin_view: Fetches the ADMIN_VIEW of the user.
      domain_public: Fetches the DOMAIN_PUBLIC view of the user.
    """
        admin_view = 0
        domain_public = 1
    customFieldMask = _messages.StringField(1)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 2, default=u'basic')
    userKey = _messages.StringField(3, required=True)
    viewType = _messages.EnumField('ViewTypeValueValuesEnum', 4, default=u'admin_view')