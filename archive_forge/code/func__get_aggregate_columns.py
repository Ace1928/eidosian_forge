import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def _get_aggregate_columns(item):
    column_map = {'metadata': 'properties'}
    hidden_columns = ['links', 'location']
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map, hidden_columns)