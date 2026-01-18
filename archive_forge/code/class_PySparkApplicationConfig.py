from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PySparkApplicationConfig(_messages.Message):
    """Represents the PySparkApplicationConfig.

  Fields:
    archiveUris: Optional. HCFS URIs of archives to be extracted into the
      working directory of each executor. Supported file types: .jar, .tar,
      .tar.gz, .tgz, and .zip.
    args: Optional. The arguments to pass to the driver. Do not include
      arguments, such as `--conf`, that can be set as job properties, since a
      collision may occur that causes an incorrect job submission.
    fileUris: Optional. HCFS URIs of files to be placed in the working
      directory of each executor. Useful for naively parallel tasks.
    jarFileUris: Optional. HCFS URIs of jar files to add to the CLASSPATHs of
      the Python driver and tasks.
    mainPythonFileUri: Required. The HCFS URI of the main Python file to use
      as the driver. Must be a .py file.
    pythonFileUris: Optional. HCFS file URIs of Python files to pass to the
      PySpark framework. Supported file types: .py, .egg, and .zip.
  """
    archiveUris = _messages.StringField(1, repeated=True)
    args = _messages.StringField(2, repeated=True)
    fileUris = _messages.StringField(3, repeated=True)
    jarFileUris = _messages.StringField(4, repeated=True)
    mainPythonFileUri = _messages.StringField(5)
    pythonFileUris = _messages.StringField(6, repeated=True)