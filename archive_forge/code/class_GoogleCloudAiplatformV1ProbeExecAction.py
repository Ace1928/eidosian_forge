from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ProbeExecAction(_messages.Message):
    """ExecAction specifies a command to execute.

  Fields:
    command: Command is the command line to execute inside the container, the
      working directory for the command is root ('/') in the container's
      filesystem. The command is simply exec'd, it is not run inside a shell,
      so traditional shell instructions ('|', etc) won't work. To use a shell,
      you need to explicitly call out to that shell. Exit status of 0 is
      treated as live/healthy and non-zero is unhealthy.
  """
    command = _messages.StringField(1, repeated=True)