from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RenderMetadata(_messages.Message):
    """RenderMetadata includes information associated with a `Release` render.

  Fields:
    cloudRun: Output only. Metadata associated with rendering for Cloud Run.
    custom: Output only. Custom metadata provided by user-defined render
      operation.
  """
    cloudRun = _messages.MessageField('CloudRunRenderMetadata', 1)
    custom = _messages.MessageField('CustomMetadata', 2)