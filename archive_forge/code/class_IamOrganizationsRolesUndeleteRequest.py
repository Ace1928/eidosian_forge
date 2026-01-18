from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsRolesUndeleteRequest(_messages.Message):
    """A IamOrganizationsRolesUndeleteRequest object.

  Fields:
    name: The `name` parameter's value depends on the target resource for the
      request, namely [`projects`](https://cloud.google.com/iam/reference/rest
      /v1/projects.roles) or [`organizations`](https://cloud.google.com/iam/re
      ference/rest/v1/organizations.roles). Each resource type's `name` value
      format is described below: * [`projects.roles.undelete()`](https://cloud
      .google.com/iam/reference/rest/v1/projects.roles/undelete):
      `projects/{PROJECT_ID}/roles/{CUSTOM_ROLE_ID}`. This method undeletes
      only [custom roles](https://cloud.google.com/iam/docs/understanding-
      custom-roles) that have been created at the project level. Example
      request URL: `https://iam.googleapis.com/v1/projects/{PROJECT_ID}/roles/
      {CUSTOM_ROLE_ID}` * [`organizations.roles.undelete()`](https://cloud.goo
      gle.com/iam/reference/rest/v1/organizations.roles/undelete):
      `organizations/{ORGANIZATION_ID}/roles/{CUSTOM_ROLE_ID}`. This method
      undeletes only [custom
      roles](https://cloud.google.com/iam/docs/understanding-custom-roles)
      that have been created at the organization level. Example request URL: `
      https://iam.googleapis.com/v1/organizations/{ORGANIZATION_ID}/roles/{CUS
      TOM_ROLE_ID}` Note: Wildcard (*) values are invalid; you must specify a
      complete project ID or organization ID.
    undeleteRoleRequest: A UndeleteRoleRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteRoleRequest = _messages.MessageField('UndeleteRoleRequest', 2)