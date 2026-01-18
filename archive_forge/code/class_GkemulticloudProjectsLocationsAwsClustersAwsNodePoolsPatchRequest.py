from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsPatchRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsPatchRequest
  object.

  Fields:
    googleCloudGkemulticloudV1AwsNodePool: A
      GoogleCloudGkemulticloudV1AwsNodePool resource to be passed as the
      request body.
    name: The name of this resource. Node pool names are formatted as
      `projects//locations//awsClusters//awsNodePools/`. For more details on
      Google Cloud resource names, see [Resource
      Names](https://cloud.google.com/apis/design/resource_names)
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field can
      only include these fields from AwsNodePool: * `annotations`. *
      `version`. * `autoscaling.min_node_count`. *
      `autoscaling.max_node_count`. * `config.config_encryption.kms_key_arn`.
      * `config.security_group_ids`. * `config.root_volume.iops`. *
      `config.root_volume.throughput`. * `config.root_volume.kms_key_arn`. *
      `config.root_volume.volume_type`. * `config.root_volume.size_gib`. *
      `config.proxy_config`. * `config.proxy_config.secret_arn`. *
      `config.proxy_config.secret_version`. * `config.ssh_config`. *
      `config.ssh_config.ec2_key_pair`. * `config.instance_placement.tenancy`.
      * `config.iam_instance_profile`. * `config.labels`. * `config.tags`. *
      `config.autoscaling_metrics_collection`. *
      `config.autoscaling_metrics_collection.granularity`. *
      `config.autoscaling_metrics_collection.metrics`. *
      `config.instance_type`. * `management.auto_repair`. * `management`. *
      `update_settings`. * `update_settings.surge_settings`. *
      `update_settings.surge_settings.max_surge`. *
      `update_settings.surge_settings.max_unavailable`.
    validateOnly: If set, only validate the request, but don't actually update
      the node pool.
  """
    googleCloudGkemulticloudV1AwsNodePool = _messages.MessageField('GoogleCloudGkemulticloudV1AwsNodePool', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)