from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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