from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkBatch(_messages.Message):
    """A configuration for running an Apache Spark (https://spark.apache.org/)
  batch workload.

  Fields:
    archiveUris: Optional. HCFS URIs of archives to be extracted into the
      working directory of each executor. Supported file types: .jar, .tar,
      .tar.gz, .tgz, and .zip.
    args: Optional. The arguments to pass to the driver. Do not include
      arguments that can be set as batch properties, such as --conf, since a
      collision can occur that causes an incorrect batch submission.
    fileUris: Optional. HCFS URIs of files to be placed in the working
      directory of each executor.
    jarFileUris: Optional. HCFS URIs of jar files to add to the classpath of
      the Spark driver and tasks.
    mainClass: Optional. The name of the driver main class. The jar file that
      contains the class must be in the classpath or specified in
      jar_file_uris.
    mainJarFileUri: Optional. The HCFS URI of the jar file that contains the
      main class.
  """
    archiveUris = _messages.StringField(1, repeated=True)
    args = _messages.StringField(2, repeated=True)
    fileUris = _messages.StringField(3, repeated=True)
    jarFileUris = _messages.StringField(4, repeated=True)
    mainClass = _messages.StringField(5)
    mainJarFileUri = _messages.StringField(6)