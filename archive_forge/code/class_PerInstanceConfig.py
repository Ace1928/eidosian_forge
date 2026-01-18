from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerInstanceConfig(_messages.Message):
    """A PerInstanceConfig object.

  Enums:
    StatusValueValuesEnum: The status of applying this per-instance
      configuration on the corresponding managed instance.

  Fields:
    fingerprint: Fingerprint of this per-instance config. This field can be
      used in optimistic locking. It is ignored when inserting a per-instance
      config. An up-to-date fingerprint must be provided in order to update an
      existing per-instance configuration or the field needs to be unset.
    name: The name of a per-instance configuration and its corresponding
      instance. Serves as a merge key during UpdatePerInstanceConfigs
      operations, that is, if a per-instance configuration with the same name
      exists then it will be updated, otherwise a new one will be created for
      the VM instance with the same name. An attempt to create a per-instance
      configconfiguration for a VM instance that either doesn't exist or is
      not part of the group will result in an error.
    preservedState: The intended preserved state for the given instance. Does
      not contain preserved state generated from a stateful policy.
    status: The status of applying this per-instance configuration on the
      corresponding managed instance.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """The status of applying this per-instance configuration on the
    corresponding managed instance.

    Values:
      APPLYING: The per-instance configuration is being applied to the
        instance, but is not yet effective, possibly waiting for the instance
        to, for example, REFRESH.
      DELETING: The per-instance configuration deletion is being applied on
        the instance, possibly waiting for the instance to, for example,
        REFRESH.
      EFFECTIVE: The per-instance configuration is effective on the instance,
        meaning that all disks, ips and metadata specified in this
        configuration are attached or set on the instance.
      NONE: *[Default]* The default status, when no per-instance configuration
        exists.
      UNAPPLIED: The per-instance configuration is set on an instance but not
        been applied yet.
      UNAPPLIED_DELETION: The per-instance configuration has been deleted, but
        the deletion is not yet applied.
    """
        APPLYING = 0
        DELETING = 1
        EFFECTIVE = 2
        NONE = 3
        UNAPPLIED = 4
        UNAPPLIED_DELETION = 5
    fingerprint = _messages.BytesField(1)
    name = _messages.StringField(2)
    preservedState = _messages.MessageField('PreservedState', 3)
    status = _messages.EnumField('StatusValueValuesEnum', 4)