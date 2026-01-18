import logging
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def _format_group_snapshot(snapshot):
    columns = ('id', 'status', 'name', 'description', 'group_id', 'group_type_id')
    column_headers = ('ID', 'Status', 'Name', 'Description', 'Group', 'Group Type')
    return (column_headers, utils.get_item_properties(snapshot, columns))