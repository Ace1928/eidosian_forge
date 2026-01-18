import logging
import uuid
from cliff import columns
import iso8601
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
def _get_server_event_columns(item, client):
    column_map = {}
    hidden_columns = ['name', 'server_id', 'links', 'location']
    if not sdk_utils.supports_microversion(client, '2.58'):
        hidden_columns.append('updated_at')
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map, hidden_columns)