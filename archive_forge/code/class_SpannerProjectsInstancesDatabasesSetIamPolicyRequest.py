from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSetIamPolicyRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSetIamPolicyRequest object.

  Fields:
    resource: REQUIRED: The Cloud Spanner resource for which the policy is
      being set. The format is `projects//instances/` for instance resources
      and `projects//instances//databases/` for databases resources.
    setIamPolicyRequest: A SetIamPolicyRequest resource to be passed as the
      request body.
  """
    resource = _messages.StringField(1, required=True)
    setIamPolicyRequest = _messages.MessageField('SetIamPolicyRequest', 2)