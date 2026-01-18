from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InvalidateCacheRequest(_messages.Message):
    """The request used by the `InvalidateCache` method.

  Fields:
    cacheTags: A list of cache tags used to identify cached objects. Cache
      tags are specified when the response is first cached, by setting the
      `Cache-Tag` response header at the origin. By default, all objects have
      a cache tag representing the HTTP status code of the response, the MIME
      content-type, and the origin. Multiple cache tags in the same
      revalidation request are treated as Boolean `OR` - for example, `tag1 OR
      tag2 OR tag3`. If a host and path are also specified, these are treated
      as Boolean `AND` with any tags. Up to 10 tags can be specified in a
      single invalidation request.
    host: The hostname to invalidate against. You can specify an exact or
      wildcard host based on the host component. For example,
      `video.example.com` or `*.example.com`.
    path: The path to invalidate against. You can specify an exact or wildcard
      path based on the a path component. For example,
      `/videos/hls/139123.mp4` or `/manifests/*`.
  """
    cacheTags = _messages.StringField(1, repeated=True)
    host = _messages.StringField(2)
    path = _messages.StringField(3)