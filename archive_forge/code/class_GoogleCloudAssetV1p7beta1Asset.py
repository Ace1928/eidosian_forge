from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1p7beta1Asset(_messages.Message):
    """An asset in Google Cloud. An asset can be any resource in the Google
  Cloud [resource hierarchy](https://cloud.google.com/resource-
  manager/docs/cloud-platform-resource-hierarchy), a resource outside the
  Google Cloud resource hierarchy (such as Google Kubernetes Engine clusters
  and objects), or a policy (e.g. IAM policy). See [Supported asset
  types](https://cloud.google.com/asset-inventory/docs/supported-asset-types)
  for more information.

  Fields:
    accessLevel: Please also refer to the [access level user
      guide](https://cloud.google.com/access-context-
      manager/docs/overview#access-levels).
    accessPolicy: Please also refer to the [access policy user
      guide](https://cloud.google.com/access-context-
      manager/docs/overview#access-policies).
    ancestors: The ancestry path of an asset in Google Cloud [resource
      hierarchy](https://cloud.google.com/resource-manager/docs/cloud-
      platform-resource-hierarchy), represented as a list of relative resource
      names. An ancestry path starts with the closest ancestor in the
      hierarchy and ends at root. If the asset is a project, folder, or
      organization, the ancestry path starts from the asset itself. Example:
      `["projects/123456789", "folders/5432", "organizations/1234"]`
    assetType: The type of the asset. Example: `compute.googleapis.com/Disk`
      See [Supported asset types](https://cloud.google.com/asset-
      inventory/docs/supported-asset-types) for more information.
    iamPolicy: A representation of the IAM policy set on a Google Cloud
      resource. There can be a maximum of one IAM policy set on any given
      resource. In addition, IAM policies inherit their granted access scope
      from any policies set on parent resources in the resource hierarchy.
      Therefore, the effectively policy is the union of both the policy set on
      this resource and each policy set on all of the resource's ancestry
      resource levels in the hierarchy. See [this
      topic](https://cloud.google.com/iam/help/allow-policies/inheritance) for
      more information.
    name: The full name of the asset. Example: `//compute.googleapis.com/proje
      cts/my_project_123/zones/zone1/instances/instance1` See [Resource names]
      (https://cloud.google.com/apis/design/resource_names#full_resource_name)
      for more information.
    orgPolicy: A representation of an [organization
      policy](https://cloud.google.com/resource-manager/docs/organization-
      policy/overview#organization_policy). There can be more than one
      organization policy with different constraints set on a given resource.
    relatedAssets: The related assets of the asset of one relationship type.
      One asset only represents one type of relationship.
    resource: A representation of the resource.
    servicePerimeter: Please also refer to the [service perimeter user
      guide](https://cloud.google.com/vpc-service-controls/docs/overview).
    updateTime: The last update timestamp of an asset. update_time is updated
      when create/update/delete operation is performed.
  """
    accessLevel = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1AccessLevel', 1)
    accessPolicy = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1AccessPolicy', 2)
    ancestors = _messages.StringField(3, repeated=True)
    assetType = _messages.StringField(4)
    iamPolicy = _messages.MessageField('Policy', 5)
    name = _messages.StringField(6)
    orgPolicy = _messages.MessageField('GoogleCloudOrgpolicyV1Policy', 7, repeated=True)
    relatedAssets = _messages.MessageField('GoogleCloudAssetV1p7beta1RelatedAssets', 8)
    resource = _messages.MessageField('GoogleCloudAssetV1p7beta1Resource', 9)
    servicePerimeter = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1ServicePerimeter', 10)
    updateTime = _messages.StringField(11)