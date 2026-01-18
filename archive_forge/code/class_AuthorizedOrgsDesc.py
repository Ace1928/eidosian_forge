from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizedOrgsDesc(_messages.Message):
    """`AuthorizedOrgsDesc` contains data for an organization's authorization
  policy.

  Enums:
    AssetTypeValueValuesEnum: The asset type of this authorized orgs desc.
      Valid values are `ASSET_TYPE_DEVICE`, and
      `ASSET_TYPE_CREDENTIAL_STRENGTH`.
    AuthorizationDirectionValueValuesEnum: The direction of the authorization
      relationship between this organization and the organizations listed in
      the `orgs` field. The valid values for this field include the following:
      `AUTHORIZATION_DIRECTION_FROM`: Allows this organization to evaluate
      traffic in the organizations listed in the `orgs` field.
      `AUTHORIZATION_DIRECTION_TO`: Allows the organizations listed in the
      `orgs` field to evaluate the traffic in this organization. For the
      authorization relationship to take effect, all of the organizations must
      authorize and specify the appropriate relationship direction. For
      example, if organization A authorized organization B and C to evaluate
      its traffic, by specifying `AUTHORIZATION_DIRECTION_TO` as the
      authorization direction, organizations B and C must specify
      `AUTHORIZATION_DIRECTION_FROM` as the authorization direction in their
      `AuthorizedOrgsDesc` resource.
    AuthorizationTypeValueValuesEnum: A granular control type for
      authorization levels. Valid value is `AUTHORIZATION_TYPE_TRUST`.

  Fields:
    assetType: The asset type of this authorized orgs desc. Valid values are
      `ASSET_TYPE_DEVICE`, and `ASSET_TYPE_CREDENTIAL_STRENGTH`.
    authorizationDirection: The direction of the authorization relationship
      between this organization and the organizations listed in the `orgs`
      field. The valid values for this field include the following:
      `AUTHORIZATION_DIRECTION_FROM`: Allows this organization to evaluate
      traffic in the organizations listed in the `orgs` field.
      `AUTHORIZATION_DIRECTION_TO`: Allows the organizations listed in the
      `orgs` field to evaluate the traffic in this organization. For the
      authorization relationship to take effect, all of the organizations must
      authorize and specify the appropriate relationship direction. For
      example, if organization A authorized organization B and C to evaluate
      its traffic, by specifying `AUTHORIZATION_DIRECTION_TO` as the
      authorization direction, organizations B and C must specify
      `AUTHORIZATION_DIRECTION_FROM` as the authorization direction in their
      `AuthorizedOrgsDesc` resource.
    authorizationType: A granular control type for authorization levels. Valid
      value is `AUTHORIZATION_TYPE_TRUST`.
    name: Resource name for the `AuthorizedOrgsDesc`. Format: `accessPolicies/
      {access_policy}/authorizedOrgsDescs/{authorized_orgs_desc}`. The
      `authorized_orgs_desc` component must begin with a letter, followed by
      alphanumeric characters or `_`. After you create an
      `AuthorizedOrgsDesc`, you cannot change its `name`.
    orgs: The list of organization ids in this AuthorizedOrgsDesc. Format:
      `organizations/` Example: `organizations/123456`
  """

    class AssetTypeValueValuesEnum(_messages.Enum):
        """The asset type of this authorized orgs desc. Valid values are
    `ASSET_TYPE_DEVICE`, and `ASSET_TYPE_CREDENTIAL_STRENGTH`.

    Values:
      ASSET_TYPE_UNSPECIFIED: No asset type specified.
      ASSET_TYPE_DEVICE: Device asset type.
      ASSET_TYPE_CREDENTIAL_STRENGTH: Credential strength asset type.
    """
        ASSET_TYPE_UNSPECIFIED = 0
        ASSET_TYPE_DEVICE = 1
        ASSET_TYPE_CREDENTIAL_STRENGTH = 2

    class AuthorizationDirectionValueValuesEnum(_messages.Enum):
        """The direction of the authorization relationship between this
    organization and the organizations listed in the `orgs` field. The valid
    values for this field include the following:
    `AUTHORIZATION_DIRECTION_FROM`: Allows this organization to evaluate
    traffic in the organizations listed in the `orgs` field.
    `AUTHORIZATION_DIRECTION_TO`: Allows the organizations listed in the
    `orgs` field to evaluate the traffic in this organization. For the
    authorization relationship to take effect, all of the organizations must
    authorize and specify the appropriate relationship direction. For example,
    if organization A authorized organization B and C to evaluate its traffic,
    by specifying `AUTHORIZATION_DIRECTION_TO` as the authorization direction,
    organizations B and C must specify `AUTHORIZATION_DIRECTION_FROM` as the
    authorization direction in their `AuthorizedOrgsDesc` resource.

    Values:
      AUTHORIZATION_DIRECTION_UNSPECIFIED: No direction specified.
      AUTHORIZATION_DIRECTION_TO: Specified orgs will evaluate traffic.
      AUTHORIZATION_DIRECTION_FROM: Specified orgs' traffic will be evaluated.
    """
        AUTHORIZATION_DIRECTION_UNSPECIFIED = 0
        AUTHORIZATION_DIRECTION_TO = 1
        AUTHORIZATION_DIRECTION_FROM = 2

    class AuthorizationTypeValueValuesEnum(_messages.Enum):
        """A granular control type for authorization levels. Valid value is
    `AUTHORIZATION_TYPE_TRUST`.

    Values:
      AUTHORIZATION_TYPE_UNSPECIFIED: No authorization type specified.
      AUTHORIZATION_TYPE_TRUST: This authorization relationship is "trust".
    """
        AUTHORIZATION_TYPE_UNSPECIFIED = 0
        AUTHORIZATION_TYPE_TRUST = 1
    assetType = _messages.EnumField('AssetTypeValueValuesEnum', 1)
    authorizationDirection = _messages.EnumField('AuthorizationDirectionValueValuesEnum', 2)
    authorizationType = _messages.EnumField('AuthorizationTypeValueValuesEnum', 3)
    name = _messages.StringField(4)
    orgs = _messages.StringField(5, repeated=True)