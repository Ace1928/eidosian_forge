from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsRolesPatchRequest(_messages.Message):
    """A IamProjectsRolesPatchRequest object.

  Fields:
    name: The `name` parameter's value depends on the target resource for the
      request, namely [`projects`](https://cloud.google.com/iam/reference/rest
      /v1/projects.roles) or [`organizations`](https://cloud.google.com/iam/re
      ference/rest/v1/organizations.roles). Each resource type's `name` value
      format is described below: * [`projects.roles.patch()`](https://cloud.go
      ogle.com/iam/reference/rest/v1/projects.roles/patch):
      `projects/{PROJECT_ID}/roles/{CUSTOM_ROLE_ID}`. This method updates only
      [custom roles](https://cloud.google.com/iam/docs/understanding-custom-
      roles) that have been created at the project level. Example request URL:
      `https://iam.googleapis.com/v1/projects/{PROJECT_ID}/roles/{CUSTOM_ROLE_
      ID}` * [`organizations.roles.patch()`](https://cloud.google.com/iam/refe
      rence/rest/v1/organizations.roles/patch):
      `organizations/{ORGANIZATION_ID}/roles/{CUSTOM_ROLE_ID}`. This method
      updates only [custom
      roles](https://cloud.google.com/iam/docs/understanding-custom-roles)
      that have been created at the organization level. Example request URL: `
      https://iam.googleapis.com/v1/organizations/{ORGANIZATION_ID}/roles/{CUS
      TOM_ROLE_ID}` Note: Wildcard (*) values are invalid; you must specify a
      complete project ID or organization ID.
    role: A Role resource to be passed as the request body.
    updateMask: A mask describing which fields in the Role have changed.
  """
    name = _messages.StringField(1, required=True)
    role = _messages.MessageField('Role', 2)
    updateMask = _messages.StringField(3)