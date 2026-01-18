from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CurrentActionValueValuesEnum(_messages.Enum):
    """[Output Only] The current action that the managed instance group has
    scheduled for the instance. Possible values: - NONE The instance is
    running, and the managed instance group does not have any scheduled
    actions for this instance. - CREATING The managed instance group is
    creating this instance. If the group fails to create this instance, it
    will try again until it is successful. - CREATING_WITHOUT_RETRIES The
    managed instance group is attempting to create this instance only once. If
    the group fails to create this instance, it does not try again and the
    group's targetSize value is decreased instead. - RECREATING The managed
    instance group is recreating this instance. - DELETING The managed
    instance group is permanently deleting this instance. - ABANDONING The
    managed instance group is abandoning this instance. The instance will be
    removed from the instance group and from any target pools that are
    associated with this group. - RESTARTING The managed instance group is
    restarting the instance. - REFRESHING The managed instance group is
    applying configuration changes to the instance without stopping it. For
    example, the group can update the target pool list for an instance without
    stopping that instance. - VERIFYING The managed instance group has created
    the instance and it is in the process of being verified.

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