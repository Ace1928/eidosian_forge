from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RoboStartingIntent(_messages.Message):
    """Message for specifying the start activities to crawl.

  Fields:
    launcherActivity: An intent that starts the main launcher activity.
    noActivity: Skips the starting activity
    startActivity: An intent that starts an activity with specific details.
    timeout: Timeout in seconds for each intent.
  """
    launcherActivity = _messages.MessageField('LauncherActivityIntent', 1)
    noActivity = _messages.MessageField('NoActivityIntent', 2)
    startActivity = _messages.MessageField('StartActivityIntent', 3)
    timeout = _messages.StringField(4)