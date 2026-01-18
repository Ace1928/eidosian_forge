from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def _get_report_format_options(self, csv_separator, csv_delimiter, csv_header, parquet):
    """Returns ReportFormatOptions instance."""
    if parquet:
        parquet_options = self.messages.ParquetOptions()
        if csv_header or csv_delimiter or csv_separator:
            raise errors.GcsApiError('CSV options cannot be set with parquet.')
        csv_options = None
    else:
        parquet_options = None
        unescaped_separator = _get_unescaped_ascii(csv_separator)
        csv_options = self.messages.CSVOptions(delimiter=csv_delimiter, headerRequired=csv_header, recordSeparator=unescaped_separator)
    return ReportFormatOptions(csv=csv_options, parquet=parquet_options)