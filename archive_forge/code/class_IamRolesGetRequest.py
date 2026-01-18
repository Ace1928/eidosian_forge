from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamRolesGetRequest(_messages.Message):
    """A IamRolesGetRequest object.

  Fields:
    name: The `name` parameter's value depends on the target resource for the
      request, namely
      [`roles`](https://cloud.google.com/iam/reference/rest/v1/roles), [`proje
      cts`](https://cloud.google.com/iam/reference/rest/v1/projects.roles), or
      [`organizations`](https://cloud.google.com/iam/reference/rest/v1/organiz
      ations.roles). Each resource type's `name` value format is described
      below: * [`roles.get()`](https://cloud.google.com/iam/reference/rest/v1/
      roles/get): `roles/{ROLE_NAME}`. This method returns results from all
      [predefined roles](https://cloud.google.com/iam/docs/understanding-
      roles#predefined_roles) in Cloud IAM. Example request URL:
      `https://iam.googleapis.com/v1/roles/{ROLE_NAME}` * [`projects.roles.get
      ()`](https://cloud.google.com/iam/reference/rest/v1/projects.roles/get):
      `projects/{PROJECT_ID}/roles/{CUSTOM_ROLE_ID}`. This method returns only
      [custom roles](https://cloud.google.com/iam/docs/understanding-custom-
      roles) that have been created at the project level. Example request URL:
      `https://iam.googleapis.com/v1/projects/{PROJECT_ID}/roles/{CUSTOM_ROLE_
      ID}` * [`organizations.roles.get()`](https://cloud.google.com/iam/refere
      nce/rest/v1/organizations.roles/get):
      `organizations/{ORGANIZATION_ID}/roles/{CUSTOM_ROLE_ID}`. This method
      returns only [custom
      roles](https://cloud.google.com/iam/docs/understanding-custom-roles)
      that have been created at the organization level. Example request URL: `
      https://iam.googleapis.com/v1/organizations/{ORGANIZATION_ID}/roles/{CUS
      TOM_ROLE_ID}` Note: Wildcard (*) values are invalid; you must specify a
      complete project ID or organization ID.
  """
    name = _messages.StringField(1, required=True)