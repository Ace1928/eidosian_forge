from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataValueValuesEnum(_messages.Enum):
    """Optional. Configures whether all, none or a subset of metadata fields
    should be added to the reported VPC flow logs. Default value is
    INCLUDE_ALL_METADATA.

    Values:
      METADATA_UNSPECIFIED: If not specified, will default to
        INCLUDE_ALL_METADATA.
      INCLUDE_ALL_METADATA: Include all metadata fields.
      EXCLUDE_ALL_METADATA: Exclude all metadata fields.
      CUSTOM_METADATA: Include only custom fields (specified in
        metadata_fields).
    """
    METADATA_UNSPECIFIED = 0
    INCLUDE_ALL_METADATA = 1
    EXCLUDE_ALL_METADATA = 2
    CUSTOM_METADATA = 3