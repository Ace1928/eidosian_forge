from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenericArtifact(_messages.Message):
    """GenericArtifact represents a generic artifact

  Fields:
    createTime: Output only. The time when the Generic module is created.
    name: Resource name of the generic artifact. project, location,
      repository, package_id and version_id create a unique generic artifact.
      i.e. "projects/test-project/locations/us-west4/repositories/test-repo/
      genericArtifacts/package_id:version_id"
    updateTime: Output only. The time when the Generic module is updated.
    version: The version of the generic artifact.
  """
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)
    updateTime = _messages.StringField(3)
    version = _messages.StringField(4)