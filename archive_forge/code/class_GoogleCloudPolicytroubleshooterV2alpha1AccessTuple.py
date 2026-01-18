from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterV2alpha1AccessTuple(_messages.Message):
    """Information about the member, resource, and permission to check.

  Fields:
    conditionContext: Optional. The Cloud IAM condition context under which
      defines the kind of access being explained. TroubleshootIamPolicy would
      explain if and why the principal has the queried permission on the
      resource specified in full_resource_name under this context
    fullResourceName: Required. The full resource name that identifies the
      resource. For example, `//compute.googleapis.com/projects/my-
      project/zones/us-central1-a/instances/my-instance`.  For examples of
      full resource names for Google Cloud services, see
      https://cloud.google.com/iam/help/troubleshooter/full-resource-names.
    permission: Required. The IAM permission to check for the specified member
      and resource.  For a complete list of IAM permissions, see
      https://cloud.google.com/iam/help/permissions/reference.  For a complete
      list of predefined IAM roles and the permissions in each role, see
      https://cloud.google.com/iam/help/roles/reference.
    principal: Required. The member, or principal, whose access you want to
      check, in the form of the email address that represents that member. For
      example, `alice@example.com` or `my-service-account@my-
      project.iam.gserviceaccount.com`.  The member must be a Google Account
      or a service account. Other types of members are not supported.
  """
    conditionContext = _messages.MessageField('GoogleCloudPolicytroubleshooterV2alpha1ConditionContext', 1)
    fullResourceName = _messages.StringField(2)
    permission = _messages.StringField(3)
    principal = _messages.StringField(4)