from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1TaskNotebookTaskConfig(_messages.Message):
    """Config for running scheduled notebooks.

  Fields:
    archiveUris: Optional. Cloud Storage URIs of archives to be extracted into
      the working directory of each executor. Supported file types: .jar,
      .tar, .tar.gz, .tgz, and .zip.
    fileUris: Optional. Cloud Storage URIs of files to be placed in the
      working directory of each executor.
    infrastructureSpec: Optional. Infrastructure specification for the
      execution.
    notebook: Required. Path to input notebook. This can be the Cloud Storage
      URI of the notebook file or the path to a Notebook Content. The
      execution args are accessible as environment variables (TASK_key=value).
  """
    archiveUris = _messages.StringField(1, repeated=True)
    fileUris = _messages.StringField(2, repeated=True)
    infrastructureSpec = _messages.MessageField('GoogleCloudDataplexV1TaskInfrastructureSpec', 3)
    notebook = _messages.StringField(4)