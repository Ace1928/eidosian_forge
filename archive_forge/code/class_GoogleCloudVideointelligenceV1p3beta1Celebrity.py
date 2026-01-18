from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1Celebrity(_messages.Message):
    """Celebrity definition.

  Fields:
    description: Textual description of additional information about the
      celebrity, if applicable.
    displayName: The celebrity name.
    name: The resource name of the celebrity. Have the format `video-
      intelligence/kg-mid` indicates a celebrity from preloaded gallery. kg-
      mid is the id in Google knowledge graph, which is unique for the
      celebrity.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    name = _messages.StringField(3)