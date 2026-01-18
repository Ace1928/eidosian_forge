from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradeUpgradeStatus(_messages.Message):
    """UpgradeStatus provides status information for each upgrade.

  Enums:
    CodeValueValuesEnum: Status code of the upgrade.

  Fields:
    code: Status code of the upgrade.
    reason: Reason for this status.
    updateTime: Last timestamp the status was updated.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Status code of the upgrade.

    Values:
      CODE_UNSPECIFIED: Required by https://linter.aip.dev/126/unspecified.
      INELIGIBLE: The upgrade is ineligible. At the scope level, this means
        the upgrade is ineligible for all the clusters in the scope.
      PENDING: The upgrade is pending. At the scope level, this means the
        upgrade is pending for all the clusters in the scope.
      IN_PROGRESS: The upgrade is in progress. At the scope level, this means
        the upgrade is in progress for at least one cluster in the scope.
      SOAKING: The upgrade has finished and is soaking until the soaking time
        is up. At the scope level, this means at least one cluster is in
        soaking while the rest are either soaking or complete.
      FORCED_SOAKING: A cluster will be forced to enter soaking if an upgrade
        doesn't finish within a certain limit, despite it's actual status.
      COMPLETE: The upgrade has passed all post conditions (soaking). At the
        scope level, this means all eligible clusters are in COMPLETE status.
    """
        CODE_UNSPECIFIED = 0
        INELIGIBLE = 1
        PENDING = 2
        IN_PROGRESS = 3
        SOAKING = 4
        FORCED_SOAKING = 5
        COMPLETE = 6
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    reason = _messages.StringField(2)
    updateTime = _messages.StringField(3)