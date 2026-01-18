from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesGetIamPolicyRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesGetIamPolicyRequest object.

  Fields:
    getIamPolicyRequest: A GetIamPolicyRequest resource to be passed as the
      request body.
    resource: REQUIRED: The Cloud Spanner resource for which the policy is
      being retrieved. The format is `projects//instances/` for instance
      resources and `projects//instances//databases/` for database resources.
  """
    getIamPolicyRequest = _messages.MessageField('GetIamPolicyRequest', 1)
    resource = _messages.StringField(2, required=True)