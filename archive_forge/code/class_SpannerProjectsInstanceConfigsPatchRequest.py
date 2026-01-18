from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstanceConfigsPatchRequest(_messages.Message):
    """A SpannerProjectsInstanceConfigsPatchRequest object.

  Fields:
    name: A unique identifier for the instance configuration. Values are of
      the form `projects//instanceConfigs/a-z*`.
    updateInstanceConfigRequest: A UpdateInstanceConfigRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateInstanceConfigRequest = _messages.MessageField('UpdateInstanceConfigRequest', 2)