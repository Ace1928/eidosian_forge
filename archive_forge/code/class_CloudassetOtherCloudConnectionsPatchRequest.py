from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetOtherCloudConnectionsPatchRequest(_messages.Message):
    """A CloudassetOtherCloudConnectionsPatchRequest object.

  Fields:
    name: Output only. Immutable. The relative resource name of an other-cloud
      connection, which is unique across Google Cloud organizations. This
      field is used to uniquely identify other-cloud connection resource. It
      contains organization number and other_cloud_connection_id when creating
      other-cloud connection. This field is immutable once resource is
      created. And currently only "aws" is allowed as the
      other_cloud_connection_id. Format: organizations/{organization_number}/o
      therCloudConnections/{other_cloud_connection_id} E.g. -
      `organizations/123/otherCloudConnections/aws`.
    otherCloudConnection: A OtherCloudConnection resource to be passed as the
      request body.
    updateMask: Required. The list of fields to update. A field represent
      symbolic field path of OtherCloudConnection. E.g.: paths:
      ["description", "collect_aws_asset_setting.qps_limit"] Note that
      `update_mask` cannot be empty, but it supports a special wildcard value
      `*`, meaning full replacement. The following immutable fields cannot be
      updated: - `name`, - `service_agent_id`, -
      `collect_aws_asset_setting.collector_role_name`, -
      `collect_aws_asset_setting.delegate_role_name`.
  """
    name = _messages.StringField(1, required=True)
    otherCloudConnection = _messages.MessageField('OtherCloudConnection', 2)
    updateMask = _messages.StringField(3)