from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceGroup(_messages.Message):
    """The resource submessage for group checks. It can be used instead of a
  monitored resource, when multiple resources are being monitored.

  Enums:
    ResourceTypeValueValuesEnum: The resource type of the group members.

  Fields:
    groupId: The group of resources being monitored. Should be only the
      [GROUP_ID], and not the full-path
      projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID].
    resourceType: The resource type of the group members.
  """

    class ResourceTypeValueValuesEnum(_messages.Enum):
        """The resource type of the group members.

    Values:
      RESOURCE_TYPE_UNSPECIFIED: Default value (not valid).
      INSTANCE: A group of instances from Google Cloud Platform (GCP) or
        Amazon Web Services (AWS).
      AWS_ELB_LOAD_BALANCER: A group of Amazon ELB load balancers.
    """
        RESOURCE_TYPE_UNSPECIFIED = 0
        INSTANCE = 1
        AWS_ELB_LOAD_BALANCER = 2
    groupId = _messages.StringField(1)
    resourceType = _messages.EnumField('ResourceTypeValueValuesEnum', 2)