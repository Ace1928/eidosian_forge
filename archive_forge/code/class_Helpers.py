from saharaclient.api import parameters as params
class Helpers(object):

    def __init__(self, sahara_client):
        self.sahara = sahara_client
        self.plugins = self.sahara.plugins

    def _get_node_processes(self, plugin):
        processes = []
        for proc_lst in plugin.node_processes.values():
            processes += proc_lst
        return [(proc_name, proc_name) for proc_name in processes]

    def get_node_processes(self, plugin_name, hadoop_version):
        plugin = self.plugins.get_version_details(plugin_name, hadoop_version)
        return self._get_node_processes(plugin)

    def _extract_parameters(self, configs, scope, applicable_target):
        parameters = []
        for config in configs:
            if config['scope'] == scope and config['applicable_target'] == applicable_target:
                parameters.append(params.Parameter(config))
        return parameters

    def get_cluster_general_configs(self, plugin_name, hadoop_version):
        plugin = self.plugins.get_version_details(plugin_name, hadoop_version)
        return self._extract_parameters(plugin.configs, 'cluster', 'general')

    def get_general_node_group_configs(self, plugin_name, hadoop_version):
        plugin = self.plugins.get_version_details(plugin_name, hadoop_version)
        return self._extract_parameters(plugin.configs, 'node', 'general')

    def get_targeted_node_group_configs(self, plugin_name, hadoop_version):
        plugin = self.plugins.get_version_details(plugin_name, hadoop_version)
        parameters = dict()
        for service in plugin.node_processes.keys():
            parameters[service] = self._extract_parameters(plugin.configs, 'node', service)
        return parameters

    def get_targeted_cluster_configs(self, plugin_name, hadoop_version):
        plugin = self.plugins.get_version_details(plugin_name, hadoop_version)
        parameters = dict()
        for service in plugin.node_processes.keys():
            parameters[service] = self._extract_parameters(plugin.configs, 'cluster', service)
        return parameters