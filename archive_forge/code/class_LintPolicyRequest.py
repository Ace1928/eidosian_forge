from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LintPolicyRequest(_messages.Message):
    """The request to lint a Cloud IAM policy object.

  Fields:
    condition: google.iam.v1.Binding.condition object to be linted.
    fullResourceName: The full resource name of the policy this lint request
      is about. The name follows the Google Cloud format for full resource
      names. For example, a Cloud project with ID `my-project` will be named
      `//cloudresourcemanager.googleapis.com/projects/my-project`. The
      resource name is not used to read a policy from IAM. Only the data in
      the request object is linted.
  """
    condition = _messages.MessageField('Expr', 1)
    fullResourceName = _messages.StringField(2)