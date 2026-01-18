from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NumericFilterOptionValueValuesEnum(_messages.Enum):
    """Required. Enum to set the option defining the datatypes numeric filter
    has to be applied to

    Values:
      NUMERIC_FILTER_OPTION_UNSPECIFIED: Numeric filter option unspecified
      NUMERIC_FILTER_OPTION_ALL: Numeric filter option that matches all
        numeric columns.
      NUMERIC_FILTER_OPTION_LIMIT: Numeric filter option that matches columns
        having numeric datatypes with specified precision and scale within the
        limited range of filter.
      NUMERIC_FILTER_OPTION_LIMITLESS: Numeric filter option that matches only
        the numeric columns with no precision and scale specified.
    """
    NUMERIC_FILTER_OPTION_UNSPECIFIED = 0
    NUMERIC_FILTER_OPTION_ALL = 1
    NUMERIC_FILTER_OPTION_LIMIT = 2
    NUMERIC_FILTER_OPTION_LIMITLESS = 3