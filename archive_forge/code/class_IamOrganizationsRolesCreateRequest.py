from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsRolesCreateRequest(_messages.Message):
    """A IamOrganizationsRolesCreateRequest object.

  Fields:
    createRoleRequest: A CreateRoleRequest resource to be passed as the
      request body.
    parent: The `parent` parameter's value depends on the target resource for
      the request, namely [`projects`](https://cloud.google.com/iam/reference/
      rest/v1/projects.roles) or [`organizations`](https://cloud.google.com/ia
      m/reference/rest/v1/organizations.roles). Each resource type's `parent`
      value format is described below: * [`projects.roles.create()`](https://c
      loud.google.com/iam/reference/rest/v1/projects.roles/create):
      `projects/{PROJECT_ID}`. This method creates project-level [custom
      roles](https://cloud.google.com/iam/docs/understanding-custom-roles).
      Example request URL:
      `https://iam.googleapis.com/v1/projects/{PROJECT_ID}/roles` * [`organiza
      tions.roles.create()`](https://cloud.google.com/iam/reference/rest/v1/or
      ganizations.roles/create): `organizations/{ORGANIZATION_ID}`. This
      method creates organization-level [custom
      roles](https://cloud.google.com/iam/docs/understanding-custom-roles).
      Example request URL:
      `https://iam.googleapis.com/v1/organizations/{ORGANIZATION_ID}/roles`
      Note: Wildcard (*) values are invalid; you must specify a complete
      project ID or organization ID.
  """
    createRoleRequest = _messages.MessageField('CreateRoleRequest', 1)
    parent = _messages.StringField(2, required=True)