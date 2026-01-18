import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
def _get_server_group_columns(item, client):
    column_map = {'member_ids': 'members'}
    hidden_columns = ['metadata', 'location']
    if sdk_utils.supports_microversion(client, '2.64'):
        hidden_columns.append('policies')
    else:
        hidden_columns.append('policy')
        hidden_columns.append('rules')
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map, hidden_columns)