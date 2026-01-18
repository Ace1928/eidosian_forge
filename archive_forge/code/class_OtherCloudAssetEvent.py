from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OtherCloudAssetEvent(_messages.Message):
    """The asset event for other-cloud asset ingestion.

  Enums:
    StateValueValuesEnum: The state of this asset.

  Fields:
    assetUri: The URI that an end-user should be able to call GET to get data
      directly from the publishers' API.
    awsInfo: For an asset fetched from AWS.
    connection: The full name of the Other-Cloud Connection resource used to
      collect this asset in the format of
      `//cloudasset.googleapis.com/organizations//OtherCloudConnections/`
      E.g.:
      cloudasset.googleapis.com/organizations/123/otherCloudConnections/aws
    contents: A representation of other-cloud asset events.
    createTime: A timestamp to represent the time when the asset was created.
      For other-cloud assets, this is optional.
    eventTime: A timestamp to represent the latest time we observe (collect)
      this resource.
    id: The identifier of this asset.
    location: The location of this asset. For AWS assets: For AWS regions, it
      is the region name listed in
      https://docs.aws.amazon.com/general/latest/gr/rande.html#regional-
      endpoints For AWS China, see `Learn more` section in
      https://docs.aws.amazon.com/general/latest/gr/rande.html#learn-more For
      AWS Gov, see GovCloud regions in:
      https://docs.aws.amazon.com/general/latest/gr/rande.html#regional-
      endpoints
    parent: The immediate parent of this asset, and it must be other-cloud
      asset. Otherwise, empty. Note: for AWS, we will populate this field only
      when the parent can be extracted from this asset's ARN.
    state: The state of this asset.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of this asset.

    Values:
      STATE_UNSPECIFIED: State is not applicable for the current asset.
      EXIST: Asset exists.
      DELETED: Asset was deleted.
    """
        STATE_UNSPECIFIED = 0
        EXIST = 1
        DELETED = 2
    assetUri = _messages.StringField(1)
    awsInfo = _messages.MessageField('AWSInfo', 2)
    connection = _messages.StringField(3)
    contents = _messages.MessageField('Content', 4, repeated=True)
    createTime = _messages.StringField(5)
    eventTime = _messages.StringField(6)
    id = _messages.MessageField('OtherCloudAssetId', 7)
    location = _messages.StringField(8)
    parent = _messages.MessageField('OtherCloudAssetId', 9)
    state = _messages.EnumField('StateValueValuesEnum', 10)