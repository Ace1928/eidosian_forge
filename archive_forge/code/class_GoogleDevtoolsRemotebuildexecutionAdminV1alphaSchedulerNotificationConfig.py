from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaSchedulerNotificationConfig(_messages.Message):
    """Defines configurations for an instance's scheduler notifications, where
  a target Pub/Sub topic will be notified whenever a task (e.g. an action or
  reservation) completes on this instance.

  Fields:
    topic: The Pub/Sub topic resource name to issue notifications to. Note
      that the topic does not need to be owned by the same project as this
      instance. Format: `projects//topics/`.
  """
    topic = _messages.StringField(1)