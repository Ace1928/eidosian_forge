from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsRolesListRequest(_messages.Message):
    """A IamOrganizationsRolesListRequest object.

  Enums:
    ViewValueValuesEnum: Optional view for the returned Role objects. When
      `FULL` is specified, the `includedPermissions` field is returned, which
      includes a list of all permissions in the role. The default value is
      `BASIC`, which does not return the `includedPermissions` field.

  Fields:
    pageSize: Optional limit on the number of roles to include in the
      response. The default is 300, and the maximum is 1,000.
    pageToken: Optional pagination token returned in an earlier
      ListRolesResponse.
    parent: The `parent` parameter's value depends on the target resource for
      the request, namely
      [`roles`](https://cloud.google.com/iam/reference/rest/v1/roles), [`proje
      cts`](https://cloud.google.com/iam/reference/rest/v1/projects.roles), or
      [`organizations`](https://cloud.google.com/iam/reference/rest/v1/organiz
      ations.roles). Each resource type's `parent` value format is described
      below: * [`roles.list()`](https://cloud.google.com/iam/reference/rest/v1
      /roles/list): An empty string. This method doesn't require a resource;
      it simply returns all [predefined
      roles](https://cloud.google.com/iam/docs/understanding-
      roles#predefined_roles) in Cloud IAM. Example request URL:
      `https://iam.googleapis.com/v1/roles` * [`projects.roles.list()`](https:
      //cloud.google.com/iam/reference/rest/v1/projects.roles/list):
      `projects/{PROJECT_ID}`. This method lists all project-level [custom
      roles](https://cloud.google.com/iam/docs/understanding-custom-roles).
      Example request URL:
      `https://iam.googleapis.com/v1/projects/{PROJECT_ID}/roles` * [`organiza
      tions.roles.list()`](https://cloud.google.com/iam/reference/rest/v1/orga
      nizations.roles/list): `organizations/{ORGANIZATION_ID}`. This method
      lists all organization-level [custom
      roles](https://cloud.google.com/iam/docs/understanding-custom-roles).
      Example request URL:
      `https://iam.googleapis.com/v1/organizations/{ORGANIZATION_ID}/roles`
      Note: Wildcard (*) values are invalid; you must specify a complete
      project ID or organization ID.
    showDeleted: Include Roles that have been deleted.
    view: Optional view for the returned Role objects. When `FULL` is
      specified, the `includedPermissions` field is returned, which includes a
      list of all permissions in the role. The default value is `BASIC`, which
      does not return the `includedPermissions` field.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional view for the returned Role objects. When `FULL` is specified,
    the `includedPermissions` field is returned, which includes a list of all
    permissions in the role. The default value is `BASIC`, which does not
    return the `includedPermissions` field.

    Values:
      BASIC: Omits the `included_permissions` field. This is the default
        value.
      FULL: Returns all fields.
    """
        BASIC = 0
        FULL = 1
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)
    view = _messages.EnumField('ViewValueValuesEnum', 5)