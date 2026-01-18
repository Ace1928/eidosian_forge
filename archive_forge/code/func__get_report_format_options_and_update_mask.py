from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def _get_report_format_options_and_update_mask(self, csv_separator, csv_delimiter, csv_header, parquet):
    """Returns a tuple of ReportFormatOptions and update_mask list."""
    report_format_options = self._get_report_format_options(csv_separator, csv_delimiter, csv_header, parquet)
    update_mask = []
    if report_format_options.parquet is not None:
        update_mask.append('parquetOptions')
    else:
        if csv_delimiter is not None:
            update_mask.append('csvOptions.delimiter')
        if csv_header is not None:
            update_mask.append('csvOptions.headerRequired')
        if csv_separator is not None:
            update_mask.append('csvOptions.recordSeparator')
    return (report_format_options, update_mask)