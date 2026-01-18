from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamPolicySearchResult(_messages.Message):
    """A result of IAM Policy search, containing information of an IAM policy.

  Fields:
    assetType: The type of the resource associated with this IAM policy.
      Example: `compute.googleapis.com/Disk`. To search against the
      `asset_type`: * specify the `asset_types` field in your search request.
    explanation: Explanation about the IAM policy search result. It contains
      additional information to explain why the search result matches the
      query.
    folders: The folder(s) that the IAM policy belongs to, in the form of
      folders/{FOLDER_NUMBER}. This field is available when the IAM policy
      belongs to one or more folders. To search against `folders`: * use a
      field query. Example: `folders:(123 OR 456)` * use a free text query.
      Example: `123` * specify the `scope` field as this folder in your search
      request.
    organization: The organization that the IAM policy belongs to, in the form
      of organizations/{ORGANIZATION_NUMBER}. This field is available when the
      IAM policy belongs to an organization. To search against `organization`:
      * use a field query. Example: `organization:123` * use a free text
      query. Example: `123` * specify the `scope` field as this organization
      in your search request.
    policy: The IAM policy directly set on the given resource. Note that the
      original IAM policy can contain multiple bindings. This only contains
      the bindings that match the given query. For queries that don't contain
      a constrain on policies (e.g., an empty query), this contains all the
      bindings. To search against the `policy` bindings: * use a field query:
      - query by the policy contained members. Example: `policy:amy@gmail.com`
      - query by the policy contained roles. Example:
      `policy:roles/compute.admin` - query by the policy contained roles'
      included permissions. Example:
      `policy.role.permissions:compute.instances.create`
    project: The project that the associated Google Cloud resource belongs to,
      in the form of projects/{PROJECT_NUMBER}. If an IAM policy is set on a
      resource (like VM instance, Cloud Storage bucket), the project field
      will indicate the project that contains the resource. If an IAM policy
      is set on a folder or orgnization, this field will be empty. To search
      against the `project`: * specify the `scope` field as this project in
      your search request.
    resource: The full resource name of the resource associated with this IAM
      policy. Example: `//compute.googleapis.com/projects/my_project_123/zones
      /zone1/instances/instance1`. See [Cloud Asset Inventory Resource Name
      Format](https://cloud.google.com/asset-inventory/docs/resource-name-
      format) for more information. To search against the `resource`: * use a
      field query. Example: `resource:organizations/123`
  """
    assetType = _messages.StringField(1)
    explanation = _messages.MessageField('Explanation', 2)
    folders = _messages.StringField(3, repeated=True)
    organization = _messages.StringField(4)
    policy = _messages.MessageField('Policy', 5)
    project = _messages.StringField(6)
    resource = _messages.StringField(7)