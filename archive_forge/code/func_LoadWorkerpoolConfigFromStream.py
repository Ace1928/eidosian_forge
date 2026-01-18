from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
def LoadWorkerpoolConfigFromStream(stream, messages, path=None):
    """Load a workerpool config file into a WorkerPool message.

  Args:
    stream: file-like object containing the JSON or YAML data to be decoded.
    messages: module, The messages module that has a WorkerPool type.
    path: str or None. Optional path to be used in error messages.

  Raises:
    ParserError: If there was a problem parsing the stream as a dict.
    ParseProtoException: If there was a problem interpreting the stream as the
      given message type.

  Returns:
    WorkerPool message, The worker pool that got decoded.
  """
    wp = cloudbuild_util.LoadMessageFromStream(stream, messages.WorkerPool, _WORKERPOOL_CONFIG_FRIENDLY_NAME, [], path)
    return wp