from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPersistentResourcesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsPersistentResourcesDeleteRequest object.

  Fields:
    name: Required. The name of the PersistentResource to be deleted. Format:
      `projects/{project}/locations/{location}/persistentResources/{persistent
      _resource}`
  """
    name = _messages.StringField(1, required=True)