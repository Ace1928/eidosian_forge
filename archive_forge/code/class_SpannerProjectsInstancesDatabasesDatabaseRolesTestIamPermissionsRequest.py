from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesDatabaseRolesTestIamPermissionsRequest(_messages.Message):
    """A
  SpannerProjectsInstancesDatabasesDatabaseRolesTestIamPermissionsRequest
  object.

  Fields:
    resource: REQUIRED: The Cloud Spanner resource for which permissions are
      being tested. The format is `projects//instances/` for instance
      resources and `projects//instances//databases/` for database resources.
    testIamPermissionsRequest: A TestIamPermissionsRequest resource to be
      passed as the request body.
  """
    resource = _messages.StringField(1, required=True)
    testIamPermissionsRequest = _messages.MessageField('TestIamPermissionsRequest', 2)