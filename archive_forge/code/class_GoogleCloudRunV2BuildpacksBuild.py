from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2BuildpacksBuild(_messages.Message):
    """Build the source using Buildpacks.

  Fields:
    baseImage: Optional. The base image used to opt into automatic base image
      updates.
    cacheImageUri: Optional. cache_image_uri is the GCR/AR URL where the cache
      image will be stored. cache_image_uri is optional and omitting it will
      disable caching. This URL must be stable across builds. It is used to
      derive a build-specific temporary URL by substituting the tag with the
      build ID. The build will clean up the temporary image on a best-effort
      basis.
    functionTarget: Optional. Name of the function target if the source is a
      function source. Required for function builds.
    runtime: The runtime name, e.g. 'go113'. Leave blank for generic builds.
  """
    baseImage = _messages.StringField(1)
    cacheImageUri = _messages.StringField(2)
    functionTarget = _messages.StringField(3)
    runtime = _messages.StringField(4)