import sys
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from saharaclient.osc import utils
def _configure_node_groups(app, node_groups, client):
    node_groups_list = dict(map(lambda x: x.split(':', 1), node_groups))
    node_groups = []
    plugins_versions = set()
    for name, count in node_groups_list.items():
        ng = utils.get_resource(client.node_group_templates, name)
        node_groups.append({'name': ng.name, 'count': int(count), 'node_group_template_id': ng.id})
        plugins_versions.add((ng.plugin_name, ng.hadoop_version))
    if len(plugins_versions) != 1:
        raise exceptions.CommandError('Node groups with the same plugins and versions must be specified')
    plugin, plugin_version = plugins_versions.pop()
    return (plugin, plugin_version, node_groups)