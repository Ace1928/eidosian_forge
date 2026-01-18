from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureViewSyncConfig(_messages.Message):
    """Configuration for Sync. Only one option is set.

  Fields:
    cron: Cron schedule (https://en.wikipedia.org/wiki/Cron) to launch
      scheduled runs. To explicitly set a timezone to the cron tab, apply a
      prefix in the cron tab: "CRON_TZ=${IANA_TIME_ZONE}" or
      "TZ=${IANA_TIME_ZONE}". The ${IANA_TIME_ZONE} may only be a valid string
      from IANA time zone database. For example, "CRON_TZ=America/New_York 1 *
      * * *", or "TZ=America/New_York 1 * * * *".
  """
    cron = _messages.StringField(1)