from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessPolicy(_messages.Message):
    """`AccessPolicy` is a container for `AccessLevels` (which define the
  necessary attributes to use Google Cloud services) and `ServicePerimeters`
  (which define regions of services able to freely pass data within a
  perimeter). An access policy is globally visible within an organization, and
  the restrictions it specifies apply to all projects within an organization.

  Fields:
    etag: Output only. An opaque identifier for the current version of the
      `AccessPolicy`. This will always be a strongly validated etag, meaning
      that two Access Polices will be identical if and only if their etags are
      identical. Clients should not expect this to be in any specific format.
    name: Resource name of the `AccessPolicy`. Format:
      `accessPolicies/{access_policy}`
    parent: Immutable. The parent of this `AccessPolicy` in the Cloud Resource
      Hierarchy Format: `organizations/{organization_id}`
    scopes: The scopes of the AccessPolicy. Scopes define which resources a
      policy can restrict and where its resources can be referenced. For
      example, policy A with `scopes=["folders/123"]` has the following
      behavior: - ServicePerimeter can only restrict projects within
      `folders/123`. - ServicePerimeter within policy A can only reference
      access levels defined within policy A. - Only one policy can include a
      given scope; thus, attempting to create a second policy which includes
      `folders/123` will result in an error. If no scopes are provided, then
      any resource within the organization can be restricted. Scopes cannot be
      modified after a policy is created. Policies can only have a single
      scope. Format: list of `folders/{folder_number}` or
      `projects/{project_number}`
    title: Required. Human readable title. Does not affect behavior.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2)
    parent = _messages.StringField(3)
    scopes = _messages.StringField(4, repeated=True)
    title = _messages.StringField(5)