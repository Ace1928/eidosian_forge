from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkRApplicationConfig(_messages.Message):
    """Represents the SparkRApplicationConfig.

  Fields:
    archiveUris: Optional. HCFS URIs of archives to be extracted into the
      working directory of each executor. Supported file types: .jar, .tar,
      .tar.gz, .tgz, and .zip.
    args: Optional. The arguments to pass to the driver. Do not include
      arguments, such as `--conf`, that can be set as job properties, since a
      collision may occur that causes an incorrect job submission.
    fileUris: Optional. HCFS URIs of files to be placed in the working
      directory of each executor. Useful for naively parallel tasks.
    mainRFileUri: Required. The HCFS URI of the main R file to use as the
      driver. Must be a .R file.
  """
    archiveUris = _messages.StringField(1, repeated=True)
    args = _messages.StringField(2, repeated=True)
    fileUris = _messages.StringField(3, repeated=True)
    mainRFileUri = _messages.StringField(4)