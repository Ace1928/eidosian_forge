from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommandValueValuesEnum(_messages.Enum):
    """The admin action; see `Command` for legal values.

    Values:
      UNSPECIFIED: Illegal value.
      BOT_UPDATE: Download and run a new version of the bot. `arg` will be a
        resource accessible via `ByteStream.Read` to obtain the new bot code.
      BOT_RESTART: Restart the bot without downloading a new version. `arg`
        will be a message to log.
      BOT_TERMINATE: Shut down the bot. `arg` will be a task resource name
        (similar to those in tasks.proto) that the bot can use to tell the
        server that it is terminating.
      HOST_RESTART: Restart the host computer. `arg` will be a message to log.
    """
    UNSPECIFIED = 0
    BOT_UPDATE = 1
    BOT_RESTART = 2
    BOT_TERMINATE = 3
    HOST_RESTART = 4