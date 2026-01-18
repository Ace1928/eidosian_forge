from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsOsPolicyAssignmentsGetRequest(_messages.Message):
    """A OsconfigProjectsLocationsOsPolicyAssignmentsGetRequest object.

  Fields:
    name: Required. The resource name of OS policy assignment. Format: `projec
      ts/{project}/locations/{location}/osPolicyAssignments/{os_policy_assignm
      ent}@{revisionId}`
  """
    name = _messages.StringField(1, required=True)