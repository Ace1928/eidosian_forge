from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChartingDimension(_messages.Message):
    """A definition for the (one) dimension column in the output. Multiple
  dimensions can be defined, but only a single column will be generated,
  containing the cross-product of the defined dimensions.

  Fields:
    column: Required. The column name within the output of the previous step
      to use.
    columnType: Optional. The type of the dimension column. This is relevant
      only if one of the bin_size fields is set. If it is empty, the type
      TIMESTAMP or INT64 will be assumed based on which bin_size field is set.
      If populated, this should be set to one of the following types: DATE,
      TIME, DATETIME, TIMESTAMP, BIGNUMERIC, INT64, NUMERIC, FLOAT64. We also
      accept all the documented aliases from
      https://cloud.google.com/bigquery/docs/reference/standard-sql/data-
      types#numeric_types as well as FLOAT (as an alias for FLOAT64).
    floatBinSize: Optional. Used for a floating-point column: FLOAT64.
    integerBinSize: Optional. Used for an integer column: INT64, NUMERIC, or
      BIGNUMERIC.
    limit: Optional. If set, any bins beyond this number will be dropped.
    limitPlusOther: Optional. If set, up to this many bins will be generated
      plus one optional additional bin. The extra bin will be named "Other"
      and will contain the sum of the (aggregated) measure points from all
      remaining bins. Setting this field will cause the dimension column type
      to be coerced to STRING if it is not already that type.
    sorting: Optional. The sorting for the dimension that defines the behavior
      of limit. If limit is not zero, this may not be set to
      SORT_ORDER_NONE.The column may be set to this dimension column or any
      measure column. If the field is empty, it will sort on the dimension
      column. If there is an anonymous measure using aggregation "count", use
      the string "*" to name it here.Note that this will not control the
      ordering of the rows in the result table in any useful way. Use the top-
      level sort ordering for that purpose.
    timeBinSize: Optional. Used for a time or date column: DATE, TIME,
      DATETIME, or TIMESTAMP. If column_type is DATE, this must be a multiple
      of 1 day. If column_type is TIME, this must be less than or equal to 24
      hours.
  """
    column = _messages.StringField(1)
    columnType = _messages.StringField(2)
    floatBinSize = _messages.FloatField(3)
    integerBinSize = _messages.IntegerField(4)
    limit = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    limitPlusOther = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    sorting = _messages.MessageField('Sorting', 7)
    timeBinSize = _messages.StringField(8)