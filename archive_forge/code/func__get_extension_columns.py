import logging
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
def _get_extension_columns(item):
    column_map = {'updated': 'updated_at'}
    hidden_columns = ['id', 'links', 'location']
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map, hidden_columns)