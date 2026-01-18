from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportDetail(_messages.Message):
    """Message describing ReportDetail object. ReportDetail represents metadata
  of generated reports for a ReportConfig. Next ID: 10

  Messages:
    LabelsValue: Labels as key value pairs

  Fields:
    labels: Labels as key value pairs
    name: Name of resource. It will be of form
      projects//locations//reportConfigs//reportDetails/.
    reportMetrics: Metrics of the report.
    reportPathPrefix: Prefix of the object name of each report's shard. This
      will have full prefix except the "extension" and "shard_id". For
      example, if the `destination_path` is `{{report-config-
      id}}/dt={{datetime}}`, the shard object name would be `gs://my-insights/
      1A34-F2E456-12B456-1C3D/dt=2022-05-20T06:35/1A34-F2E456-12B456-
      1C3D_2022-05-20T06:35_5.csv` and the value of `report_path_prefix` field
      would be `gs://my-insights/1A34-F2E456-12B456-1C3D/dt=2022-05-
      20T06:35/1A34-F2E456-12B456-1C3D_2022-05-20T06:35_`.
    shardsCount: Total shards generated for the report.
    snapshotTime: The snapshot time. All the report data is referenced at this
      point of time.
    status: Status of the ReportDetail.
    targetDatetime: The date for which report is generated. The time part of
      target_datetime will be zero till we support multiple reports per day.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key value pairs

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    labels = _messages.MessageField('LabelsValue', 1)
    name = _messages.StringField(2)
    reportMetrics = _messages.MessageField('Metrics', 3)
    reportPathPrefix = _messages.StringField(4)
    shardsCount = _messages.IntegerField(5)
    snapshotTime = _messages.StringField(6)
    status = _messages.MessageField('Status', 7)
    targetDatetime = _messages.MessageField('DateTime', 8)