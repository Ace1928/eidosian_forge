from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NotebookIdleShutdownConfig(_messages.Message):
    """The idle shutdown configuration of NotebookRuntimeTemplate, which
  contains the idle_timeout as required field.

  Fields:
    idleShutdownDisabled: Whether Idle Shutdown is disabled in this
      NotebookRuntimeTemplate.
    idleTimeout: Required. Duration is accurate to the second. In Notebook,
      Idle Timeout is accurate to minute so the range of idle_timeout (second)
      is: 10 * 60 ~ 1440 * 60.
  """
    idleShutdownDisabled = _messages.BooleanField(1)
    idleTimeout = _messages.StringField(2)