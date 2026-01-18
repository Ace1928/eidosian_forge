from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstanceConfigsGetRequest(_messages.Message):
    """A SpannerProjectsInstanceConfigsGetRequest object.

  Fields:
    name: Required. The name of the requested instance configuration. Values
      are of the form `projects//instanceConfigs/`.
  """
    name = _messages.StringField(1, required=True)