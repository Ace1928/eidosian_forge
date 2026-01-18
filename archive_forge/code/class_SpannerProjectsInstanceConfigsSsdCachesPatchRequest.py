from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstanceConfigsSsdCachesPatchRequest(_messages.Message):
    """A SpannerProjectsInstanceConfigsSsdCachesPatchRequest object.

  Fields:
    name: A unique identifier for the cache. Values are of the form
      `projects//instanceConfigs//ssdCaches/a-z*[a-z0-9]`. The final segment
      of the name must be between 2 and 64 characters in length. A cache's
      name cannot be changed after the cache is created.
    updateSsdCacheRequest: A UpdateSsdCacheRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    updateSsdCacheRequest = _messages.MessageField('UpdateSsdCacheRequest', 2)