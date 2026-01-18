from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsModulesCreateRequest(_messages.Message):
    """A MediaassetProjectsLocationsModulesCreateRequest object.

  Fields:
    module: A Module resource to be passed as the request body.
    moduleId: The ID of the module resource to be created.
    parent: Required. The parent resource name, in the following form:
      `projects/{project}/locations/{location}`.
  """
    module = _messages.MessageField('Module', 1)
    moduleId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)