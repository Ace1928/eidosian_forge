from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsBuildsSubmitRequest(_messages.Message):
    """A RunProjectsLocationsBuildsSubmitRequest object.

  Fields:
    googleCloudRunV2SubmitBuildRequest: A GoogleCloudRunV2SubmitBuildRequest
      resource to be passed as the request body.
    parent: Required. The project and location to build in. Location must be a
      region, e.g., 'us-central1' or 'global' if the global builder is to be
      used. Format: projects/{project}/locations/{location}
  """
    googleCloudRunV2SubmitBuildRequest = _messages.MessageField('GoogleCloudRunV2SubmitBuildRequest', 1)
    parent = _messages.StringField(2, required=True)