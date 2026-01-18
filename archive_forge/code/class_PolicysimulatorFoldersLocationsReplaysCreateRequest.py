from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicysimulatorFoldersLocationsReplaysCreateRequest(_messages.Message):
    """A PolicysimulatorFoldersLocationsReplaysCreateRequest object.

  Fields:
    googleCloudPolicysimulatorV1betaReplay: A
      GoogleCloudPolicysimulatorV1betaReplay resource to be passed as the
      request body.
    parent: Required. The parent resource where this Replay will be created.
      This resource must be a project, folder, or organization with a
      location. Example: `projects/my-example-project/locations/global`
  """
    googleCloudPolicysimulatorV1betaReplay = _messages.MessageField('GoogleCloudPolicysimulatorV1betaReplay', 1)
    parent = _messages.StringField(2, required=True)