from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsRolesDeleteRequest(_messages.Message):
    """A IamProjectsRolesDeleteRequest object.

  Fields:
    etag: Used to perform a consistent read-modify-write.
    name: The `name` parameter's value depends on the target resource for the
      request, namely [`projects`](https://cloud.google.com/iam/reference/rest
      /v1/projects.roles) or [`organizations`](https://cloud.google.com/iam/re
      ference/rest/v1/organizations.roles). Each resource type's `name` value
      format is described below: * [`projects.roles.delete()`](https://cloud.g
      oogle.com/iam/reference/rest/v1/projects.roles/delete):
      `projects/{PROJECT_ID}/roles/{CUSTOM_ROLE_ID}`. This method deletes only
      [custom roles](https://cloud.google.com/iam/docs/understanding-custom-
      roles) that have been created at the project level. Example request URL:
      `https://iam.googleapis.com/v1/projects/{PROJECT_ID}/roles/{CUSTOM_ROLE_
      ID}` * [`organizations.roles.delete()`](https://cloud.google.com/iam/ref
      erence/rest/v1/organizations.roles/delete):
      `organizations/{ORGANIZATION_ID}/roles/{CUSTOM_ROLE_ID}`. This method
      deletes only [custom
      roles](https://cloud.google.com/iam/docs/understanding-custom-roles)
      that have been created at the organization level. Example request URL: `
      https://iam.googleapis.com/v1/organizations/{ORGANIZATION_ID}/roles/{CUS
      TOM_ROLE_ID}` Note: Wildcard (*) values are invalid; you must specify a
      complete project ID or organization ID.
  """
    etag = _messages.BytesField(1)
    name = _messages.StringField(2, required=True)