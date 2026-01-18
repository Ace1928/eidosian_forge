from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersWatchRequest(_messages.Message):
    """A DirectoryUsersWatchRequest object.

  Enums:
    EventValueValuesEnum: Event on which subscription is intended (if
      subscribing)
    OrderByValueValuesEnum: Column to use for sorting results
    ProjectionValueValuesEnum: What subset of fields to fetch for this user.
    SortOrderValueValuesEnum: Whether to return results in ascending or
      descending order.
    ViewTypeValueValuesEnum: Whether to fetch the ADMIN_VIEW or DOMAIN_PUBLIC
      view of the user.

  Fields:
    channel: A Channel resource to be passed as the request body.
    customFieldMask: Comma-separated list of schema names. All fields from
      these schemas are fetched. This should only be set when
      projection=custom.
    customer: Immutable ID of the G Suite account. In case of multi-domain, to
      fetch all users for a customer, fill this field instead of domain.
    domain: Name of the domain. Fill this field to get users from only this
      domain. To return all users in a multi-domain fill customer field
      instead.
    event: Event on which subscription is intended (if subscribing)
    maxResults: Maximum number of results to return.
    orderBy: Column to use for sorting results
    pageToken: Token to specify next page in the list
    projection: What subset of fields to fetch for this user.
    query: Query string search. Should be of the form "". Complete
      documentation is at https://developers.google.com/admin-
      sdk/directory/v1/guides/search-users
    showDeleted: If set to true, retrieves the list of deleted users.
      (Default: false)
    sortOrder: Whether to return results in ascending or descending order.
    viewType: Whether to fetch the ADMIN_VIEW or DOMAIN_PUBLIC view of the
      user.
  """

    class EventValueValuesEnum(_messages.Enum):
        """Event on which subscription is intended (if subscribing)

    Values:
      add: User Created Event
      delete: User Deleted Event
      makeAdmin: User Admin Status Change Event
      undelete: User Undeleted Event
      update: User Updated Event
    """
        add = 0
        delete = 1
        makeAdmin = 2
        undelete = 3
        update = 4

    class OrderByValueValuesEnum(_messages.Enum):
        """Column to use for sorting results

    Values:
      email: Primary email of the user.
      familyName: User's family name.
      givenName: User's given name.
    """
        email = 0
        familyName = 1
        givenName = 2

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

    class SortOrderValueValuesEnum(_messages.Enum):
        """Whether to return results in ascending or descending order.

    Values:
      ASCENDING: Ascending order.
      DESCENDING: Descending order.
    """
        ASCENDING = 0
        DESCENDING = 1

    class ViewTypeValueValuesEnum(_messages.Enum):
        """Whether to fetch the ADMIN_VIEW or DOMAIN_PUBLIC view of the user.

    Values:
      admin_view: Fetches the ADMIN_VIEW of the user.
      domain_public: Fetches the DOMAIN_PUBLIC view of the user.
    """
        admin_view = 0
        domain_public = 1
    channel = _messages.MessageField('Channel', 1)
    customFieldMask = _messages.StringField(2)
    customer = _messages.StringField(3)
    domain = _messages.StringField(4)
    event = _messages.EnumField('EventValueValuesEnum', 5)
    maxResults = _messages.IntegerField(6, variant=_messages.Variant.INT32, default=100)
    orderBy = _messages.EnumField('OrderByValueValuesEnum', 7)
    pageToken = _messages.StringField(8)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 9, default=u'basic')
    query = _messages.StringField(10)
    showDeleted = _messages.StringField(11)
    sortOrder = _messages.EnumField('SortOrderValueValuesEnum', 12)
    viewType = _messages.EnumField('ViewTypeValueValuesEnum', 13, default=u'admin_view')