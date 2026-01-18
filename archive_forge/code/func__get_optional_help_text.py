from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _get_optional_help_text(require_create_flags, flag_name):
    """Returns a text to be added for create command's help text."""
    optional_text_map = {'destination': ' Defaults to <SOURCE_BUCKET_URL>/inventory_reports/.', 'metadata_fields': ' Defaults to all fields being included.', 'start_date': ' Defaults to tomorrow.', 'end_date': ' Defaults to one year from --schedule-starts value.', 'frequency': ' Defaults to DAILY.'}
    return optional_text_map[flag_name] if require_create_flags else ''