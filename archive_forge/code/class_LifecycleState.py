from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LifecycleState(_messages.Message):
    """Describes the lifecycle state of an Immersive Stream for XR resource.

  Enums:
    StateValueValuesEnum: Current lifecycle state of the resource (e.g. if
      it's Live or Deprecated).

  Fields:
    description: Human readable message describing details about the current
      state.
    state: Current lifecycle state of the resource (e.g. if it's Live or
      Deprecated).
  """

    class StateValueValuesEnum(_messages.Enum):
        """Current lifecycle state of the resource (e.g. if it's Live or
    Deprecated).

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      READY: Resource is ready and available for use.
      IN_USE: Resource is being used (referenced by other resources). In order
        to delete the resource, it must go through deprecation process to
        ensure it's no longer in use by other resources.
      CREATING: Resource is being created.
      UPDATING: Resource is being updated.
      DELETING: Resource is being deleted.
      ERROR: Resource encountered an error and is in indeterministic state.
    """
        STATE_UNSPECIFIED = 0
        READY = 1
        IN_USE = 2
        CREATING = 3
        UPDATING = 4
        DELETING = 5
        ERROR = 6
    description = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)