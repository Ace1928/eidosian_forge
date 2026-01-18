from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionInstanceGroupManagersApplyUpdatesRequest(_messages.Message):
    """RegionInstanceGroupManagers.applyUpdatesToInstances

  Enums:
    MinimalActionValueValuesEnum: The minimal action that you want to perform
      on each instance during the update: - REPLACE: At minimum, delete the
      instance and create it again. - RESTART: Stop the instance and start it
      again. - REFRESH: Do not stop the instance and limit disruption as much
      as possible. - NONE: Do not disrupt the instance at all. By default, the
      minimum action is NONE. If your update requires a more disruptive action
      than you set with this flag, the necessary action is performed to
      execute the update.
    MostDisruptiveAllowedActionValueValuesEnum: The most disruptive action
      that you want to perform on each instance during the update: - REPLACE:
      Delete the instance and create it again. - RESTART: Stop the instance
      and start it again. - REFRESH: Do not stop the instance and limit
      disruption as much as possible. - NONE: Do not disrupt the instance at
      all. By default, the most disruptive allowed action is REPLACE. If your
      update requires a more disruptive action than you set with this flag,
      the update request will fail.

  Fields:
    allInstances: Flag to update all instances instead of specified list of
      "instances". If the flag is set to true then the instances may not be
      specified in the request.
    instances: The list of URLs of one or more instances for which you want to
      apply updates. Each URL can be a full URL or a partial URL, such as
      zones/[ZONE]/instances/[INSTANCE_NAME].
    minimalAction: The minimal action that you want to perform on each
      instance during the update: - REPLACE: At minimum, delete the instance
      and create it again. - RESTART: Stop the instance and start it again. -
      REFRESH: Do not stop the instance and limit disruption as much as
      possible. - NONE: Do not disrupt the instance at all. By default, the
      minimum action is NONE. If your update requires a more disruptive action
      than you set with this flag, the necessary action is performed to
      execute the update.
    mostDisruptiveAllowedAction: The most disruptive action that you want to
      perform on each instance during the update: - REPLACE: Delete the
      instance and create it again. - RESTART: Stop the instance and start it
      again. - REFRESH: Do not stop the instance and limit disruption as much
      as possible. - NONE: Do not disrupt the instance at all. By default, the
      most disruptive allowed action is REPLACE. If your update requires a
      more disruptive action than you set with this flag, the update request
      will fail.
  """

    class MinimalActionValueValuesEnum(_messages.Enum):
        """The minimal action that you want to perform on each instance during
    the update: - REPLACE: At minimum, delete the instance and create it
    again. - RESTART: Stop the instance and start it again. - REFRESH: Do not
    stop the instance and limit disruption as much as possible. - NONE: Do not
    disrupt the instance at all. By default, the minimum action is NONE. If
    your update requires a more disruptive action than you set with this flag,
    the necessary action is performed to execute the update.

    Values:
      NONE: Do not perform any action.
      REFRESH: Do not stop the instance.
      REPLACE: (Default.) Replace the instance according to the replacement
        method option.
      RESTART: Stop the instance and start it again.
    """
        NONE = 0
        REFRESH = 1
        REPLACE = 2
        RESTART = 3

    class MostDisruptiveAllowedActionValueValuesEnum(_messages.Enum):
        """The most disruptive action that you want to perform on each instance
    during the update: - REPLACE: Delete the instance and create it again. -
    RESTART: Stop the instance and start it again. - REFRESH: Do not stop the
    instance and limit disruption as much as possible. - NONE: Do not disrupt
    the instance at all. By default, the most disruptive allowed action is
    REPLACE. If your update requires a more disruptive action than you set
    with this flag, the update request will fail.

    Values:
      NONE: Do not perform any action.
      REFRESH: Do not stop the instance.
      REPLACE: (Default.) Replace the instance according to the replacement
        method option.
      RESTART: Stop the instance and start it again.
    """
        NONE = 0
        REFRESH = 1
        REPLACE = 2
        RESTART = 3
    allInstances = _messages.BooleanField(1)
    instances = _messages.StringField(2, repeated=True)
    minimalAction = _messages.EnumField('MinimalActionValueValuesEnum', 3)
    mostDisruptiveAllowedAction = _messages.EnumField('MostDisruptiveAllowedActionValueValuesEnum', 4)