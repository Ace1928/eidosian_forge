from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FilterTypeValueValuesEnum(_messages.Enum):
    """The specified filter type

    Values:
      FILTER_TYPE_UNSPECIFIED: Filter type is unspecified. This is not valid
        in a well-formed request.
      RESOURCE_LABEL: Filter on a resource label value
      METRIC_LABEL: Filter on a metrics label value
      USER_METADATA_LABEL: Filter on a user metadata label value
      SYSTEM_METADATA_LABEL: Filter on a system metadata label value
      GROUP: Filter on a group id
    """
    FILTER_TYPE_UNSPECIFIED = 0
    RESOURCE_LABEL = 1
    METRIC_LABEL = 2
    USER_METADATA_LABEL = 3
    SYSTEM_METADATA_LABEL = 4
    GROUP = 5