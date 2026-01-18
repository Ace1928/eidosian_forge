from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemoteworkersV1test2CommandOutputs(_messages.Message):
    """DEPRECATED - use CommandResult instead. Describes the actual outputs
  from the task.

  Fields:
    exitCode: exit_code is only fully reliable if the status' code is OK. If
      the task exceeded its deadline or was cancelled, the process may still
      produce an exit code as it is cancelled, and this will be populated, but
      a successful (zero) is unlikely to be correct unless the status code is
      OK.
    outputs: The output files. The blob referenced by the digest should
      contain one of the following (implementation-dependent): * A marshalled
      DirectoryMetadata of the returned filesystem * A LUCI-style .isolated
      file
  """
    exitCode = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    outputs = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2Digest', 2)