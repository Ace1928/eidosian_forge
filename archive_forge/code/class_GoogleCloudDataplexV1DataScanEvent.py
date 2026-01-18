from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataScanEvent(_messages.Message):
    """These messages contain information about the execution of a datascan.
  The monitored resource is 'DataScan' Next ID: 13

  Enums:
    ScopeValueValuesEnum: The scope of the data scan (e.g. full, incremental).
    StateValueValuesEnum: The status of the data scan job.
    TriggerValueValuesEnum: The trigger type of the data scan job.
    TypeValueValuesEnum: The type of the data scan.

  Fields:
    createTime: The time when the data scan job was created.
    dataProfile: Data profile result for data profile type data scan.
    dataProfileConfigs: Applied configs for data profile type data scan.
    dataQuality: Data quality result for data quality type data scan.
    dataQualityConfigs: Applied configs for data quality type data scan.
    dataSource: The data source of the data scan
    endTime: The time when the data scan job finished.
    jobId: The identifier of the specific data scan job this log entry is for.
    message: The message describing the data scan job event.
    postScanActionsResult: The result of post scan actions.
    scope: The scope of the data scan (e.g. full, incremental).
    specVersion: A version identifier of the spec which was used to execute
      this job.
    startTime: The time when the data scan job started to run.
    state: The status of the data scan job.
    trigger: The trigger type of the data scan job.
    type: The type of the data scan.
  """

    class ScopeValueValuesEnum(_messages.Enum):
        """The scope of the data scan (e.g. full, incremental).

    Values:
      SCOPE_UNSPECIFIED: An unspecified scope type.
      FULL: Data scan runs on all of the data.
      INCREMENTAL: Data scan runs on incremental data.
    """
        SCOPE_UNSPECIFIED = 0
        FULL = 1
        INCREMENTAL = 2

    class StateValueValuesEnum(_messages.Enum):
        """The status of the data scan job.

    Values:
      STATE_UNSPECIFIED: Unspecified job state.
      STARTED: Data scan job started.
      SUCCEEDED: Data scan job successfully completed.
      FAILED: Data scan job was unsuccessful.
      CANCELLED: Data scan job was cancelled.
      CREATED: Data scan job was createed.
    """
        STATE_UNSPECIFIED = 0
        STARTED = 1
        SUCCEEDED = 2
        FAILED = 3
        CANCELLED = 4
        CREATED = 5

    class TriggerValueValuesEnum(_messages.Enum):
        """The trigger type of the data scan job.

    Values:
      TRIGGER_UNSPECIFIED: An unspecified trigger type.
      ON_DEMAND: Data scan triggers on demand.
      SCHEDULE: Data scan triggers as per schedule.
    """
        TRIGGER_UNSPECIFIED = 0
        ON_DEMAND = 1
        SCHEDULE = 2

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the data scan.

    Values:
      SCAN_TYPE_UNSPECIFIED: An unspecified data scan type.
      DATA_PROFILE: Data scan for data profile.
      DATA_QUALITY: Data scan for data quality.
    """
        SCAN_TYPE_UNSPECIFIED = 0
        DATA_PROFILE = 1
        DATA_QUALITY = 2
    createTime = _messages.StringField(1)
    dataProfile = _messages.MessageField('GoogleCloudDataplexV1DataScanEventDataProfileResult', 2)
    dataProfileConfigs = _messages.MessageField('GoogleCloudDataplexV1DataScanEventDataProfileAppliedConfigs', 3)
    dataQuality = _messages.MessageField('GoogleCloudDataplexV1DataScanEventDataQualityResult', 4)
    dataQualityConfigs = _messages.MessageField('GoogleCloudDataplexV1DataScanEventDataQualityAppliedConfigs', 5)
    dataSource = _messages.StringField(6)
    endTime = _messages.StringField(7)
    jobId = _messages.StringField(8)
    message = _messages.StringField(9)
    postScanActionsResult = _messages.MessageField('GoogleCloudDataplexV1DataScanEventPostScanActionsResult', 10)
    scope = _messages.EnumField('ScopeValueValuesEnum', 11)
    specVersion = _messages.StringField(12)
    startTime = _messages.StringField(13)
    state = _messages.EnumField('StateValueValuesEnum', 14)
    trigger = _messages.EnumField('TriggerValueValuesEnum', 15)
    type = _messages.EnumField('TypeValueValuesEnum', 16)