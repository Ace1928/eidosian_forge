from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsInsightsConfiguredInsightRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsInsightsConfiguredInsightRequest object.

  Enums:
    AggregationValueValuesEnum: Required. Aggregation type. Available
      aggregation could be fetched by calling insight list and get APIs in
      `BASIC` view.

  Fields:
    aggregation: Required. Aggregation type. Available aggregation could be
      fetched by calling insight list and get APIs in `BASIC` view.
    customGrouping_fieldFilter: Optional. Filterable parameters to be added to
      the grouping clause. Available fields could be fetched by calling
      insight list and get APIs in `BASIC` view. `=` is the only comparison
      operator supported. `AND` is the only logical operator supported. Usage:
      field_filter="fieldName1=fieldVal1 AND fieldName2=fieldVal2". NOTE: Only
      `AND` conditions are allowed. NOTE: Use the `filter_alias` from
      `Insight.Metadata.Field` message for the filtering the corresponding
      fields in this filter field. (These expressions are based on the filter
      language described at https://google.aip.dev/160).
    customGrouping_groupFields: Required. Fields to be used for grouping.
      NOTE: Use the `filter_alias` from `Insight.Metadata.Field` message for
      declaring the fields to be grouped-by here.
    endTime: Required. Ending time for the duration for which insight is to be
      pulled.
    fieldFilter: Optional. Other filterable/configurable parameters as
      applicable to the selected insight. Available fields could be fetched by
      calling insight list and get APIs in `BASIC` view. `=` is the only
      comparison operator supported. `AND` is the only logical operator
      supported. Usage: field_filter="fieldName1=fieldVal1 AND
      fieldName2=fieldVal2". NOTE: Only `AND` conditions are allowed. NOTE:
      Use the `filter_alias` from `Insight.Metadata.Field` message for the
      filtering the corresponding fields in this filter field. (These
      expressions are based on the filter language described at
      https://google.aip.dev/160).
    group: Optional. Group id of the available groupings for the insight.
      Available groupings could be fetched by calling insight list and get
      APIs in `BASIC` view.
    insight: Required. The resource name of the insight using the form: `organ
      izations/{organization_id}/locations/{location_id}/insights/{insight_id}
      ` `projects/{project_id}/locations/{location_id}/insights/{insight_id}`.
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. Used to fetch the page represented by the token.
      Fetches the first page when not set.
    startTime: Required. Starting time for the duration for which insight is
      to be pulled.
  """

    class AggregationValueValuesEnum(_messages.Enum):
        """Required. Aggregation type. Available aggregation could be fetched by
    calling insight list and get APIs in `BASIC` view.

    Values:
      AGGREGATION_UNSPECIFIED: Unspecified.
      HOURLY: Insight should be aggregated at hourly level.
      DAILY: Insight should be aggregated at daily level.
      WEEKLY: Insight should be aggregated at weekly level.
      MONTHLY: Insight should be aggregated at monthly level.
      CUSTOM_DATE_RANGE: Insight should be aggregated at the custom date range
        passed in as the start and end time in the request.
    """
        AGGREGATION_UNSPECIFIED = 0
        HOURLY = 1
        DAILY = 2
        WEEKLY = 3
        MONTHLY = 4
        CUSTOM_DATE_RANGE = 5
    aggregation = _messages.EnumField('AggregationValueValuesEnum', 1)
    customGrouping_fieldFilter = _messages.StringField(2)
    customGrouping_groupFields = _messages.StringField(3, repeated=True)
    endTime = _messages.StringField(4)
    fieldFilter = _messages.StringField(5)
    group = _messages.StringField(6)
    insight = _messages.StringField(7, required=True)
    pageSize = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(9)
    startTime = _messages.StringField(10)