from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def _ParseUpdateMode(alloydb_messages, update_mode):
    if update_mode:
        return alloydb_messages.UpdatePolicy.ModeValueValuesEnum.lookup_by_name(update_mode.upper())
    return None