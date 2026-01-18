from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstancesUpdateRequest(_messages.Message):
    """A ComputeInstancesUpdateRequest object.

  Enums:
    MinimalActionValueValuesEnum: Specifies the action to take when updating
      an instance even if the updated properties do not require it. If not
      specified, then Compute Engine acts based on the minimum action that the
      updated properties require.
    MostDisruptiveAllowedActionValueValuesEnum: Specifies the most disruptive
      action that can be taken on the instance as part of the update. Compute
      Engine returns an error if the instance properties require a more
      disruptive action as part of the instance update. Valid options from
      lowest to highest are NO_EFFECT, REFRESH, and RESTART.

  Fields:
    instance: Name of the instance resource to update.
    instanceResource: A Instance resource to be passed as the request body.
    minimalAction: Specifies the action to take when updating an instance even
      if the updated properties do not require it. If not specified, then
      Compute Engine acts based on the minimum action that the updated
      properties require.
    mostDisruptiveAllowedAction: Specifies the most disruptive action that can
      be taken on the instance as part of the update. Compute Engine returns
      an error if the instance properties require a more disruptive action as
      part of the instance update. Valid options from lowest to highest are
      NO_EFFECT, REFRESH, and RESTART.
    project: Project ID for this request.
    requestId: An optional request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. For example,
      consider a situation where you make an initial request and the request
      times out. If you make the request again with the same request ID, the
      server can check if original operation with the same request ID was
      received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      ( 00000000-0000-0000-0000-000000000000).
    zone: The name of the zone for this request.
  """

    class MinimalActionValueValuesEnum(_messages.Enum):
        """Specifies the action to take when updating an instance even if the
    updated properties do not require it. If not specified, then Compute
    Engine acts based on the minimum action that the updated properties
    require.

    Values:
      INVALID: <no description>
      NO_EFFECT: No changes can be made to the instance.
      REFRESH: The instance will not restart.
      RESTART: The instance will restart.
    """
        INVALID = 0
        NO_EFFECT = 1
        REFRESH = 2
        RESTART = 3

    class MostDisruptiveAllowedActionValueValuesEnum(_messages.Enum):
        """Specifies the most disruptive action that can be taken on the instance
    as part of the update. Compute Engine returns an error if the instance
    properties require a more disruptive action as part of the instance
    update. Valid options from lowest to highest are NO_EFFECT, REFRESH, and
    RESTART.

    Values:
      INVALID: <no description>
      NO_EFFECT: No changes can be made to the instance.
      REFRESH: The instance will not restart.
      RESTART: The instance will restart.
    """
        INVALID = 0
        NO_EFFECT = 1
        REFRESH = 2
        RESTART = 3
    instance = _messages.StringField(1, required=True)
    instanceResource = _messages.MessageField('Instance', 2)
    minimalAction = _messages.EnumField('MinimalActionValueValuesEnum', 3)
    mostDisruptiveAllowedAction = _messages.EnumField('MostDisruptiveAllowedActionValueValuesEnum', 4)
    project = _messages.StringField(5, required=True)
    requestId = _messages.StringField(6)
    zone = _messages.StringField(7, required=True)