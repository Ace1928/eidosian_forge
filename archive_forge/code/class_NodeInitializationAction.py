from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeInitializationAction(_messages.Message):
    """Specifies an executable to run on a fully configured node and a timeout
  period for executable completion.

  Fields:
    executableFile: Required. Cloud Storage URI of executable file.
    executionTimeout: Optional. Amount of time executable has to complete.
      Default is 10 minutes (see JSON representation of Duration
      (https://developers.google.com/protocol-
      buffers/docs/proto3#json)).Cluster creation fails with an explanatory
      error message (the name of the executable that caused the error and the
      exceeded timeout period) if the executable is not completed at end of
      the timeout period.
  """
    executableFile = _messages.StringField(1)
    executionTimeout = _messages.StringField(2)