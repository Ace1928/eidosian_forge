from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
def LoadWorkerpoolConfigFromPath(path, messages):
    """Load a workerpool config file into a WorkerPool message.

  Args:
    path: str. Path to the JSON or YAML data to be decoded.
    messages: module, The messages module that has a WorkerPool type.

  Raises:
    files.MissingFileError: If the file does not exist.
    ParserError: If there was a problem parsing the file as a dict.
    ParseProtoException: If there was a problem interpreting the file as the
      given message type.

  Returns:
    WorkerPool message, The worker pool that got decoded.
  """
    wp = cloudbuild_util.LoadMessageFromPath(path, messages.WorkerPool, _WORKERPOOL_CONFIG_FRIENDLY_NAME)
    return wp