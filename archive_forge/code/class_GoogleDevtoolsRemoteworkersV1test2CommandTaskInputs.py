from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemoteworkersV1test2CommandTaskInputs(_messages.Message):
    """Describes the inputs to a shell-style task.

  Fields:
    arguments: The command itself to run (e.g., argv). This field should be
      passed directly to the underlying operating system, and so it must be
      sensible to that operating system. For example, on Windows, the first
      argument might be "C:\\Windows\\System32\\ping.exe" - that is, using drive
      letters and backslashes. A command for a *nix system, on the other hand,
      would use forward slashes. All other fields in the RWAPI must
      consistently use forward slashes, since those fields may be interpretted
      by both the service and the bot.
    environmentVariables: All environment variables required by the task.
    files: The input filesystem to be set up prior to the task beginning. The
      contents should be a repeated set of FileMetadata messages though other
      formats are allowed if better for the implementation (eg, a LUCI-style
      .isolated file). This field is repeated since implementations might want
      to cache the metadata, in which case it may be useful to break up
      portions of the filesystem that change frequently (eg, specific input
      files) from those that don't (eg, standard header files).
    inlineBlobs: Inline contents for blobs expected to be needed by the bot to
      execute the task. For example, contents of entries in `files` or blobs
      that are indirectly referenced by an entry there. The bot should check
      against this list before downloading required task inputs to reduce the
      number of communications between itself and the remote CAS server.
    workingDirectory: Directory from which a command is executed. It is a
      relative directory with respect to the bot's working directory (i.e.,
      "./"). If it is non-empty, then it must exist under "./". Otherwise,
      "./" will be used.
  """
    arguments = _messages.StringField(1, repeated=True)
    environmentVariables = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2CommandTaskInputsEnvironmentVariable', 2, repeated=True)
    files = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2Digest', 3, repeated=True)
    inlineBlobs = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2Blob', 4, repeated=True)
    workingDirectory = _messages.StringField(5)