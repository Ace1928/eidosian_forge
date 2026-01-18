from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSaasplatformInsightsV1alphaCustomGrouping(_messages.Message):
    """Customised grouping option that allows setting the group_by fields and
  also the filters togather for a configured insight request.

  Fields:
    fieldFilter: Optional. Filterable parameters to be added to the grouping
      clause. Available fields could be fetched by calling insight list and
      get APIs in `BASIC` view. `=` is the only comparison operator supported.
      `AND` is the only logical operator supported. Usage:
      field_filter="fieldName1=fieldVal1 AND fieldName2=fieldVal2". NOTE: Only
      `AND` conditions are allowed. NOTE: Use the `filter_alias` from
      `Insight.Metadata.Field` message for the filtering the corresponding
      fields in this filter field. (These expressions are based on the filter
      language described at https://google.aip.dev/160).
    groupFields: Required. Fields to be used for grouping. NOTE: Use the
      `filter_alias` from `Insight.Metadata.Field` message for declaring the
      fields to be grouped-by here.
  """
    fieldFilter = _messages.StringField(1)
    groupFields = _messages.StringField(2, repeated=True)