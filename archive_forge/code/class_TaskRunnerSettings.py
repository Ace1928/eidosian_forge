from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskRunnerSettings(_messages.Message):
    """Taskrunner configuration settings.

  Fields:
    alsologtostderr: Whether to also send taskrunner log info to stderr.
    baseTaskDir: The location on the worker for task-specific subdirectories.
    baseUrl: The base URL for the taskrunner to use when accessing Google
      Cloud APIs. When workers access Google Cloud APIs, they logically do so
      via relative URLs. If this field is specified, it supplies the base URL
      to use for resolving these relative URLs. The normative algorithm used
      is defined by RFC 1808, "Relative Uniform Resource Locators". If not
      specified, the default value is "http://www.googleapis.com/"
    commandlinesFileName: The file to store preprocessing commands in.
    continueOnException: Whether to continue taskrunner if an exception is
      hit.
    dataflowApiVersion: The API version of endpoint, e.g. "v1b3"
    harnessCommand: The command to launch the worker harness.
    languageHint: The suggested backend language.
    logDir: The directory on the VM to store logs.
    logToSerialconsole: Whether to send taskrunner log info to Google Compute
      Engine VM serial console.
    logUploadLocation: Indicates where to put logs. If this is not specified,
      the logs will not be uploaded. The supported resource type is: Google
      Cloud Storage: storage.googleapis.com/{bucket}/{object}
      bucket.storage.googleapis.com/{object}
    oauthScopes: The OAuth2 scopes to be requested by the taskrunner in order
      to access the Cloud Dataflow API.
    parallelWorkerSettings: The settings to pass to the parallel worker
      harness.
    streamingWorkerMainClass: The streaming worker main class name.
    taskGroup: The UNIX group ID on the worker VM to use for tasks launched by
      taskrunner; e.g. "wheel".
    taskUser: The UNIX user ID on the worker VM to use for tasks launched by
      taskrunner; e.g. "root".
    tempStoragePrefix: The prefix of the resources the taskrunner should use
      for temporary storage. The supported resource type is: Google Cloud
      Storage: storage.googleapis.com/{bucket}/{object}
      bucket.storage.googleapis.com/{object}
    vmId: The ID string of the VM.
    workflowFileName: The file to store the workflow in.
  """
    alsologtostderr = _messages.BooleanField(1)
    baseTaskDir = _messages.StringField(2)
    baseUrl = _messages.StringField(3)
    commandlinesFileName = _messages.StringField(4)
    continueOnException = _messages.BooleanField(5)
    dataflowApiVersion = _messages.StringField(6)
    harnessCommand = _messages.StringField(7)
    languageHint = _messages.StringField(8)
    logDir = _messages.StringField(9)
    logToSerialconsole = _messages.BooleanField(10)
    logUploadLocation = _messages.StringField(11)
    oauthScopes = _messages.StringField(12, repeated=True)
    parallelWorkerSettings = _messages.MessageField('WorkerSettings', 13)
    streamingWorkerMainClass = _messages.StringField(14)
    taskGroup = _messages.StringField(15)
    taskUser = _messages.StringField(16)
    tempStoragePrefix = _messages.StringField(17)
    vmId = _messages.StringField(18)
    workflowFileName = _messages.StringField(19)