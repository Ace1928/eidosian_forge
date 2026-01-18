from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyBinding(_messages.Message):
    """Grouping of IAM role and IAM member.

  Fields:
    member: Required. Member to bind the role with. See
      /iam/docs/reference/rest/v1/Policy#Binding for how to format each
      member. Eg. - user:myuser@mydomain.com - serviceAccount:my-service-
      account@app.gserviceaccount.com
    role: Required. Role to apply. Only allowlisted roles can be used at the
      specified granularity. The role must be one of the following: -
      'roles/container.hostServiceAgentUser' applied on the shared VPC host
      project - 'roles/compute.securityAdmin' applied on the shared VPC host
      project - 'roles/compute.networkAdmin' applied on the shared VPC host
      project - 'roles/compute.xpnAdmin' applied on the shared VPC host
      project - 'roles/dns.admin' applied on the shared VPC host project
  """
    member = _messages.StringField(1)
    role = _messages.StringField(2)