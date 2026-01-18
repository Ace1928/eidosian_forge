from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2LogFile(_messages.Message):
    """A `LogFile` is a log stored in the CAS.

  Fields:
    digest: The digest of the log contents.
    humanReadable: This is a hint as to the purpose of the log, and is set to
      true if the log is human-readable text that can be usefully displayed to
      a user, and false otherwise. For instance, if a command-line client
      wishes to print the server logs to the terminal for a failed action,
      this allows it to avoid displaying a binary file.
  """
    digest = _messages.MessageField('BuildBazelRemoteExecutionV2Digest', 1)
    humanReadable = _messages.BooleanField(2)