from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsSpecialistPoolsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsSpecialistPoolsDeleteRequest object.

  Fields:
    force: If set to true, any specialist managers in this SpecialistPool will
      also be deleted. (Otherwise, the request will only work if the
      SpecialistPool has no specialist managers.)
    name: Required. The resource name of the SpecialistPool to delete. Format:
      `projects/{project}/locations/{location}/specialistPools/{specialist_poo
      l}`
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)