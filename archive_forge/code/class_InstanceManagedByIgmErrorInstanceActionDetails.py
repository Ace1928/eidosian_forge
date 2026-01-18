from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceManagedByIgmErrorInstanceActionDetails(_messages.Message):
    """A InstanceManagedByIgmErrorInstanceActionDetails object.

  Enums:
    ActionValueValuesEnum: [Output Only] Action that managed instance group
      was executing on the instance when the error occurred. Possible values:

  Fields:
    action: [Output Only] Action that managed instance group was executing on
      the instance when the error occurred. Possible values:
    instance: [Output Only] The URL of the instance. The URL can be set even
      if the instance has not yet been created.
    version: [Output Only] Version this instance was created from, or was
      being created from, but the creation failed. Corresponds to one of the
      versions that were set on the Instance Group Manager resource at the
      time this instance was being created.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """[Output Only] Action that managed instance group was executing on the
    instance when the error occurred. Possible values:

    Values:
      ABANDONING: The managed instance group is abandoning this instance. The
        instance will be removed from the instance group and from any target
        pools that are associated with this group.
      CREATING: The managed instance group is creating this instance. If the
        group fails to create this instance, it will try again until it is
        successful.
      CREATING_WITHOUT_RETRIES: The managed instance group is attempting to
        create this instance only once. If the group fails to create this
        instance, it does not try again and the group's targetSize value is
        decreased.
      DELETING: The managed instance group is permanently deleting this
        instance.
      NONE: The managed instance group has not scheduled any actions for this
        instance.
      RECREATING: The managed instance group is recreating this instance.
      REFRESHING: The managed instance group is applying configuration changes
        to the instance without stopping it. For example, the group can update
        the target pool list for an instance without stopping that instance.
      RESTARTING: The managed instance group is restarting this instance.
      RESUMING: The managed instance group is resuming this instance.
      STARTING: The managed instance group is starting this instance.
      STOPPING: The managed instance group is stopping this instance.
      SUSPENDING: The managed instance group is suspending this instance.
      VERIFYING: The managed instance group is verifying this already created
        instance. Verification happens every time the instance is (re)created
        or restarted and consists of: 1. Waiting until health check specified
        as part of this managed instance group's autohealing policy reports
        HEALTHY. Note: Applies only if autohealing policy has a health check
        specified 2. Waiting for addition verification steps performed as
        post-instance creation (subject to future extensions).
    """
        ABANDONING = 0
        CREATING = 1
        CREATING_WITHOUT_RETRIES = 2
        DELETING = 3
        NONE = 4
        RECREATING = 5
        REFRESHING = 6
        RESTARTING = 7
        RESUMING = 8
        STARTING = 9
        STOPPING = 10
        SUSPENDING = 11
        VERIFYING = 12
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    instance = _messages.StringField(2)
    version = _messages.MessageField('ManagedInstanceVersion', 3)