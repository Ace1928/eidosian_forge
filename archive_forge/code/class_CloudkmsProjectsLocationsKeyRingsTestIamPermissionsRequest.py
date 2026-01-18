from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsTestIamPermissionsRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsTestIamPermissionsRequest object.

  Fields:
    resource: REQUIRED: The resource for which the policy detail is being
      requested. See the operation documentation for the appropriate value for
      this field.
    testIamPermissionsRequest: A TestIamPermissionsRequest resource to be
      passed as the request body.
  """
    resource = _messages.StringField(1, required=True)
    testIamPermissionsRequest = _messages.MessageField('TestIamPermissionsRequest', 2)