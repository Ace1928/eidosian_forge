from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.audit_manager import audit_scopes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.audit_manager import exception_utils
from googlecloudsdk.command_lib.audit_manager import flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def SaveReport(self, response, output_directory, output_file_name):
    """Save the generated scope."""
    is_empty_directory_path = output_directory == ''
    directory_path = '' if is_empty_directory_path else output_directory + '/'
    file_path = directory_path + output_file_name + _FILE_EXTENSION
    content_bytes = response.scopeReportContents
    files.WriteBinaryFileContents(file_path, content_bytes, overwrite=False)