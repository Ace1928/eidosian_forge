from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPackagesTagsCreateRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesPackagesTagsCreateRequest
  object.

  Fields:
    parent: The name of the parent resource where the tag will be created.
    tag: A Tag resource to be passed as the request body.
    tagId: The tag id to use for this repository.
  """
    parent = _messages.StringField(1, required=True)
    tag = _messages.MessageField('Tag', 2)
    tagId = _messages.StringField(3)