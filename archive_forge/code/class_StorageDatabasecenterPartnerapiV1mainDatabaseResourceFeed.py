from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDatabasecenterPartnerapiV1mainDatabaseResourceFeed(_messages.Message):
    """DatabaseResourceFeed is the top level proto to be used to ingest
  different database resource level events into Condor platform.

  Enums:
    FeedTypeValueValuesEnum: Required. Type feed to be ingested into condor

  Fields:
    feedTimestamp: Required. Timestamp when feed is generated.
    feedType: Required. Type feed to be ingested into condor
    recommendationSignalData: More feed data would be added in subsequent CLs
    resourceHealthSignalData: A
      StorageDatabasecenterPartnerapiV1mainDatabaseResourceHealthSignalData
      attribute.
    resourceId: Primary key associated with the Resource. resource_id is
      available in individual feed level as well.
    resourceMetadata: A
      StorageDatabasecenterPartnerapiV1mainDatabaseResourceMetadata attribute.
  """

    class FeedTypeValueValuesEnum(_messages.Enum):
        """Required. Type feed to be ingested into condor

    Values:
      FEEDTYPE_UNSPECIFIED: <no description>
      RESOURCE_METADATA: Database resource metadata feed from control plane
      OBSERVABILITY_DATA: Database resource monitoring data
      SECURITY_FINDING_DATA: Database resource security health signal data
      RECOMMENDATION_SIGNAL_DATA: Database resource recommendation signal data
    """
        FEEDTYPE_UNSPECIFIED = 0
        RESOURCE_METADATA = 1
        OBSERVABILITY_DATA = 2
        SECURITY_FINDING_DATA = 3
        RECOMMENDATION_SIGNAL_DATA = 4
    feedTimestamp = _messages.StringField(1)
    feedType = _messages.EnumField('FeedTypeValueValuesEnum', 2)
    recommendationSignalData = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainDatabaseResourceRecommendationSignalData', 3)
    resourceHealthSignalData = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainDatabaseResourceHealthSignalData', 4)
    resourceId = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainDatabaseResourceId', 5)
    resourceMetadata = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainDatabaseResourceMetadata', 6)