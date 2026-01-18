from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserOrganization(_messages.Message):
    """JSON template for an organization entry.

  Fields:
    costCenter: The cost center of the users department.
    customType: Custom type.
    department: Department within the organization.
    description: Description of the organization.
    domain: The domain to which the organization belongs to.
    fullTimeEquivalent: The full-time equivalent millipercent within the
      organization (100000 = 100%).
    location: Location of the organization. This need not be fully qualified
      address.
    name: Name of the organization
    primary: If it user's primary organization.
    symbol: Symbol of the organization.
    title: Title (designation) of the user in the organization.
    type: Each entry can have a type which indicates standard types of that
      entry. For example organization could be of school, work etc. In
      addition to the standard type, an entry can have a custom type and can
      give it any name. Such types should have the CUSTOM value as type and
      also have a CustomType value.
  """
    costCenter = _messages.StringField(1)
    customType = _messages.StringField(2)
    department = _messages.StringField(3)
    description = _messages.StringField(4)
    domain = _messages.StringField(5)
    fullTimeEquivalent = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    location = _messages.StringField(7)
    name = _messages.StringField(8)
    primary = _messages.BooleanField(9)
    symbol = _messages.StringField(10)
    title = _messages.StringField(11)
    type = _messages.StringField(12)