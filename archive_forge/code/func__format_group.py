import argparse
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def _format_group(group):
    columns = ('id', 'status', 'name', 'description', 'group_type', 'volume_types', 'availability_zone', 'created_at', 'volumes', 'group_snapshot_id', 'source_group_id')
    column_headers = ('ID', 'Status', 'Name', 'Description', 'Group Type', 'Volume Types', 'Availability Zone', 'Created At', 'Volumes', 'Group Snapshot ID', 'Source Group ID')
    return (column_headers, utils.get_item_properties(group, columns))