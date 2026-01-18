from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsOsPolicyAssignmentsCreateRequest(_messages.Message):
    """A OsconfigProjectsLocationsOsPolicyAssignmentsCreateRequest object.

  Fields:
    oSPolicyAssignment: A OSPolicyAssignment resource to be passed as the
      request body.
    osPolicyAssignmentId: Required. The logical name of the OS policy
      assignment in the project with the following restrictions: * Must
      contain only lowercase letters, numbers, and hyphens. * Must start with
      a letter. * Must be between 1-63 characters. * Must end with a number or
      a letter. * Must be unique within the project.
    parent: Required. The parent resource name in the form:
      projects/{project}/locations/{location}
  """
    oSPolicyAssignment = _messages.MessageField('OSPolicyAssignment', 1)
    osPolicyAssignmentId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)