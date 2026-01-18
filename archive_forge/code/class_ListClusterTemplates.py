from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import cluster_templates as ct_v1
class ListClusterTemplates(ct_v1.ListClusterTemplates):
    """Lists cluster templates"""
    log = logging.getLogger(__name__ + '.ListClusterTemplates')

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        search_opts = {}
        if parsed_args.plugin:
            search_opts['plugin_name'] = parsed_args.plugin
        if parsed_args.plugin_version:
            search_opts['plugin_version'] = parsed_args.plugin_version
        data = client.cluster_templates.list(search_opts=search_opts)
        if parsed_args.name:
            data = utils.get_by_name_substring(data, parsed_args.name)
        if parsed_args.long:
            columns = ('name', 'id', 'plugin_name', 'plugin_version', 'node_groups', 'description')
            column_headers = utils.prepare_column_headers(columns)
        else:
            columns = ('name', 'id', 'plugin_name', 'plugin_version')
            column_headers = utils.prepare_column_headers(columns)
        return (column_headers, (osc_utils.get_item_properties(s, columns, formatters={'node_groups': ct_v1._format_node_groups_list}) for s in data))