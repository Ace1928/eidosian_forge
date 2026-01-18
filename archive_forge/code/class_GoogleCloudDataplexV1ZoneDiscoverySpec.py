from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ZoneDiscoverySpec(_messages.Message):
    """Settings to manage the metadata discovery and publishing in a zone.

  Fields:
    csvOptions: Optional. Configuration for CSV data.
    enabled: Required. Whether discovery is enabled.
    excludePatterns: Optional. The list of patterns to apply for selecting
      data to exclude during discovery. For Cloud Storage bucket assets, these
      are interpreted as glob patterns used to match object names. For
      BigQuery dataset assets, these are interpreted as patterns to match
      table names.
    includePatterns: Optional. The list of patterns to apply for selecting
      data to include during discovery if only a subset of the data should
      considered. For Cloud Storage bucket assets, these are interpreted as
      glob patterns used to match object names. For BigQuery dataset assets,
      these are interpreted as patterns to match table names.
    jsonOptions: Optional. Configuration for Json data.
    schedule: Optional. Cron schedule (https://en.wikipedia.org/wiki/Cron) for
      running discovery periodically. Successive discovery runs must be
      scheduled at least 60 minutes apart. The default value is to run
      discovery every 60 minutes. To explicitly set a timezone to the cron
      tab, apply a prefix in the cron tab: "CRON_TZ=${IANA_TIME_ZONE}" or
      TZ=${IANA_TIME_ZONE}". The ${IANA_TIME_ZONE} may only be a valid string
      from IANA time zone database. For example, CRON_TZ=America/New_York 1 *
      * * *, or TZ=America/New_York 1 * * * *.
  """
    csvOptions = _messages.MessageField('GoogleCloudDataplexV1ZoneDiscoverySpecCsvOptions', 1)
    enabled = _messages.BooleanField(2)
    excludePatterns = _messages.StringField(3, repeated=True)
    includePatterns = _messages.StringField(4, repeated=True)
    jsonOptions = _messages.MessageField('GoogleCloudDataplexV1ZoneDiscoverySpecJsonOptions', 5)
    schedule = _messages.StringField(6)