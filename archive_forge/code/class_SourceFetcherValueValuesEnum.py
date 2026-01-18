from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceFetcherValueValuesEnum(_messages.Enum):
    """Optional. Option to specify the tool to fetch the source file for the
    build.

    Values:
      SOURCE_FETCHER_UNSPECIFIED: Unspecified defaults to GSUTIL.
      GSUTIL: Use the "gsutil" tool to download the source file.
      GCS_FETCHER: Use the Cloud Storage Fetcher tool to download the source
        file.
    """
    SOURCE_FETCHER_UNSPECIFIED = 0
    GSUTIL = 1
    GCS_FETCHER = 2