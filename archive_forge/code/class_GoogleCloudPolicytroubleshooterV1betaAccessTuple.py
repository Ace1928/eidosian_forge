from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudPolicytroubleshooterV1betaAccessTuple(_messages.Message):
    """Information about the member, resource, and permission to check.

  Fields:
    fullResourceName: Required. The full resource name that identifies the
      resource. For example, `//compute.googleapis.com/projects/my-
      project/zones/us-central1-a/instances/my-instance`. For examples of full
      resource names for Google Cloud services, see
      https://cloud.google.com/iam/help/troubleshooter/full-resource-names.
    permission: Required. The IAM permission to check for the specified member
      and resource. For a complete list of IAM permissions, see
      https://cloud.google.com/iam/help/permissions/reference. For a complete
      list of predefined IAM roles and the permissions in each role, see
      https://cloud.google.com/iam/help/roles/reference.
    principal: Required. The member, or principal, whose access you want to
      check, in the form of the email address that represents that member. For
      example, `alice@example.com` or `my-service-account@my-
      project.iam.gserviceaccount.com`. The member must be a Google Account or
      a service account. Other types of members are not supported.
  """
    fullResourceName = _messages.StringField(1)
    permission = _messages.StringField(2)
    principal = _messages.StringField(3)