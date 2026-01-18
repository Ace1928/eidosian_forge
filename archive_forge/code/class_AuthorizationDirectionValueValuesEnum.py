from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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