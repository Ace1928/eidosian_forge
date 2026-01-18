from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildbotCommandDurations(_messages.Message):
    """CommandDuration contains the various duration metrics tracked when a bot
  performs a command.

  Fields:
    casRelease: The time spent to release the CAS blobs used by the task.
    cmWaitForAssignment: The time spent waiting for Container Manager to
      assign an asynchronous container for execution.
    dockerPrep: The time spent preparing the command to be run in a Docker
      container (includes pulling the Docker image, if necessary).
    dockerPrepStartTime: The timestamp when docker preparation begins.
    download: The time spent downloading the input files and constructing the
      working directory.
    downloadStartTime: The timestamp when downloading the input files begins.
    execStartTime: The timestamp when execution begins.
    execution: The time spent executing the command (i.e., doing useful work).
    isoPrepDone: The timestamp when preparation is done and bot starts
      downloading files.
    overall: The time spent completing the command, in total.
    stderr: The time spent uploading the stderr logs.
    stdout: The time spent uploading the stdout logs.
    upload: The time spent uploading the output files.
    uploadStartTime: The timestamp when uploading the output files begins.
  """
    casRelease = _messages.StringField(1)
    cmWaitForAssignment = _messages.StringField(2)
    dockerPrep = _messages.StringField(3)
    dockerPrepStartTime = _messages.StringField(4)
    download = _messages.StringField(5)
    downloadStartTime = _messages.StringField(6)
    execStartTime = _messages.StringField(7)
    execution = _messages.StringField(8)
    isoPrepDone = _messages.StringField(9)
    overall = _messages.StringField(10)
    stderr = _messages.StringField(11)
    stdout = _messages.StringField(12)
    upload = _messages.StringField(13)
    uploadStartTime = _messages.StringField(14)