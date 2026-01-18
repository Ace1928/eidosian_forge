from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DynamicGroupQuery(_messages.Message):
    """Defines a query on a resource.

  Enums:
    ResourceTypeValueValuesEnum: Resource type for the Dynamic Group Query

  Fields:
    query: Query that determines the memberships of the dynamic group.
      Examples: All users with at least one `organizations.department` of
      engineering. `user.organizations.exists(org,
      org.department=='engineering')` All users with at least one location
      that has `area` of `foo` and `building_id` of `bar`.
      `user.locations.exists(loc, loc.area=='foo' && loc.building_id=='bar')`
      All users with any variation of the name John Doe (case-insensitive
      queries add `equalsIgnoreCase()` to the value being queried).
      `user.name.value.equalsIgnoreCase('jOhn DoE')`
    resourceType: Resource type for the Dynamic Group Query
  """

    class ResourceTypeValueValuesEnum(_messages.Enum):
        """Resource type for the Dynamic Group Query

    Values:
      RESOURCE_TYPE_UNSPECIFIED: Default value (not valid)
      USER: For queries on User
    """
        RESOURCE_TYPE_UNSPECIFIED = 0
        USER = 1
    query = _messages.StringField(1)
    resourceType = _messages.EnumField('ResourceTypeValueValuesEnum', 2)