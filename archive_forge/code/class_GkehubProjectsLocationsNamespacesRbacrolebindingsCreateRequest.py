from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsNamespacesRbacrolebindingsCreateRequest(_messages.Message):
    """A GkehubProjectsLocationsNamespacesRbacrolebindingsCreateRequest object.

  Fields:
    parent: Required. The parent (project and location) where the
      RBACRoleBinding will be created. Specified in the format
      `projects/*/locations/*/namespaces/*`.
    rBACRoleBinding: A RBACRoleBinding resource to be passed as the request
      body.
    rbacrolebindingId: Required. Client chosen ID for the RBACRoleBinding.
      `rbacrolebinding_id` must be a valid RFC 1123 compliant DNS label: 1. At
      most 63 characters in length 2. It must consist of lower case
      alphanumeric characters or `-` 3. It must start and end with an
      alphanumeric character Which can be expressed as the regex:
      `[a-z0-9]([-a-z0-9]*[a-z0-9])?`, with a maximum length of 63 characters.
  """
    parent = _messages.StringField(1, required=True)
    rBACRoleBinding = _messages.MessageField('RBACRoleBinding', 2)
    rbacrolebindingId = _messages.StringField(3)