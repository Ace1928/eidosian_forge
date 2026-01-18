import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def _list_take_action(self, client, app, parsed_args):
    search_opts = {}
    if parsed_args.plugin:
        search_opts['plugin_name'] = parsed_args.plugin
    if parsed_args.plugin_version:
        search_opts['hadoop_version'] = parsed_args.plugin_version
    data = client.node_group_templates.list(search_opts=search_opts)
    if parsed_args.name:
        data = get_by_name_substring(data, parsed_args.name)
    if app.api_version['data_processing'] == '2':
        if parsed_args.long:
            columns = ('name', 'id', 'plugin_name', 'plugin_version', 'node_processes', 'description')
            column_headers = prepare_column_headers(columns)
        else:
            columns = ('name', 'id', 'plugin_name', 'plugin_version')
            column_headers = prepare_column_headers(columns)
    elif parsed_args.long:
        columns = ('name', 'id', 'plugin_name', 'hadoop_version', 'node_processes', 'description')
        column_headers = prepare_column_headers(columns, {'hadoop_version': 'plugin_version'})
    else:
        columns = ('name', 'id', 'plugin_name', 'hadoop_version')
        column_headers = prepare_column_headers(columns, {'hadoop_version': 'plugin_version'})
    return (column_headers, (osc_utils.get_item_properties(s, columns, formatters={'node_processes': osc_utils.format_list}) for s in data))