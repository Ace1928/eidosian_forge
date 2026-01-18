from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Runnable(_messages.Message):
    """Runnable describes instructions for executing a specific script or
  container as part of a Task.

  Messages:
    LabelsValue: Labels for this Runnable.

  Fields:
    alwaysRun: By default, after a Runnable fails, no further Runnable are
      executed. This flag indicates that this Runnable must be run even if the
      Task has already failed. This is useful for Runnables that copy output
      files off of the VM or for debugging. The always_run flag does not
      override the Task's overall max_run_duration. If the max_run_duration
      has expired then no further Runnables will execute, not even always_run
      Runnables.
    background: This flag allows a Runnable to continue running in the
      background while the Task executes subsequent Runnables. This is useful
      to provide services to other Runnables (or to provide debugging support
      tools like SSH servers).
    barrier: Barrier runnable.
    container: Container runnable.
    displayName: Optional. DisplayName is an optional field that can be
      provided by the caller. If provided, it will be used in logs and other
      outputs to identify the script, making it easier for users to understand
      the logs. If not provided the index of the runnable will be used for
      outputs.
    environment: Environment variables for this Runnable (overrides variables
      set for the whole Task or TaskGroup).
    ignoreExitStatus: Normally, a non-zero exit status causes the Task to
      fail. This flag allows execution of other Runnables to continue instead.
    labels: Labels for this Runnable.
    script: Script runnable.
    timeout: Timeout for this Runnable.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels for this Runnable.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    alwaysRun = _messages.BooleanField(1)
    background = _messages.BooleanField(2)
    barrier = _messages.MessageField('Barrier', 3)
    container = _messages.MessageField('Container', 4)
    displayName = _messages.StringField(5)
    environment = _messages.MessageField('Environment', 6)
    ignoreExitStatus = _messages.BooleanField(7)
    labels = _messages.MessageField('LabelsValue', 8)
    script = _messages.MessageField('Script', 9)
    timeout = _messages.StringField(10)