from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Waiter(_messages.Message):
    """A Waiter resource waits for some end condition within a RuntimeConfig
  resource to be met before it returns. For example, assume you have a
  distributed system where each node writes to a Variable resource indicating
  the node's readiness as part of the startup process. You then configure a
  Waiter resource with the success condition set to wait until some number of
  nodes have checked in. Afterwards, your application runs some arbitrary code
  after the condition has been met and the waiter returns successfully. Once
  created, a Waiter resource is immutable. To learn more about using waiters,
  read the [Creating a Waiter](/deployment-manager/runtime-
  configurator/creating-a-waiter) documentation.

  Fields:
    createTime: Output only. The instant at which this Waiter resource was
      created. Adding the value of `timeout` to this instant yields the
      timeout deadline for the waiter.
    done: Output only. If the value is `false`, it means the waiter is still
      waiting for one of its conditions to be met. If true, the waiter has
      finished. If the waiter finished due to a timeout or failure, `error`
      will be set.
    error: Output only. If the waiter ended due to a failure or timeout, this
      value will be set.
    failure: [Optional] The failure condition of this waiter. If this
      condition is met, `done` will be set to `true` and the `error` code will
      be set to `ABORTED`. The failure condition takes precedence over the
      success condition. If both conditions are met, a failure will be
      indicated. This value is optional; if no failure condition is set, the
      only failure scenario will be a timeout.
    name: The name of the Waiter resource, in the format:
      projects/[PROJECT_ID]/configs/[CONFIG_NAME]/waiters/[WAITER_NAME] The
      `[PROJECT_ID]` must be a valid Google Cloud project ID, the
      `[CONFIG_NAME]` must be a valid RuntimeConfig resource, the
      `[WAITER_NAME]` must match RFC 1035 segment specification, and the
      length of `[WAITER_NAME]` must be less than 64 bytes. After you create a
      Waiter resource, you cannot change the resource name.
    success: [Required] The success condition. If this condition is met,
      `done` will be set to `true` and the `error` value will remain unset.
      The failure condition takes precedence over the success condition. If
      both conditions are met, a failure will be indicated.
    timeout: [Required] Specifies the timeout of the waiter in seconds,
      beginning from the instant that `waiters().create` method is called. If
      this time elapses before the success or failure conditions are met, the
      waiter fails and sets the `error` code to `DEADLINE_EXCEEDED`.
  """
    createTime = _messages.StringField(1)
    done = _messages.BooleanField(2)
    error = _messages.MessageField('Status', 3)
    failure = _messages.MessageField('EndCondition', 4)
    name = _messages.StringField(5)
    success = _messages.MessageField('EndCondition', 6)
    timeout = _messages.StringField(7)