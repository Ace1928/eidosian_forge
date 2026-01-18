from saharaclient.api import base
class ClusterTemplateManagerV2(ClusterTemplateManagerV1):
    NotUpdated = base.NotUpdated()

    def create(self, name, plugin_name, plugin_version, description=None, cluster_configs=None, node_groups=None, anti_affinity=None, net_id=None, default_image_id=None, use_autoconfig=None, shares=None, is_public=None, is_protected=None, domain_name=None):
        """Create a Cluster Template."""
        data = {'name': name, 'plugin_name': plugin_name, 'plugin_version': plugin_version}
        return self._do_create(data, description, cluster_configs, node_groups, anti_affinity, net_id, default_image_id, use_autoconfig, shares, is_public, is_protected, domain_name)

    def update(self, cluster_template_id, name=NotUpdated, plugin_name=NotUpdated, plugin_version=NotUpdated, description=NotUpdated, cluster_configs=NotUpdated, node_groups=NotUpdated, anti_affinity=NotUpdated, net_id=NotUpdated, default_image_id=NotUpdated, use_autoconfig=NotUpdated, shares=NotUpdated, is_public=NotUpdated, is_protected=NotUpdated, domain_name=NotUpdated):
        """Update a Cluster Template."""
        data = {}
        self._copy_if_updated(data, name=name, plugin_name=plugin_name, plugin_version=plugin_version, description=description, cluster_configs=cluster_configs, node_groups=node_groups, anti_affinity=anti_affinity, neutron_management_network=net_id, default_image_id=default_image_id, use_autoconfig=use_autoconfig, shares=shares, is_public=is_public, is_protected=is_protected, domain_name=domain_name)
        return self._patch('/cluster-templates/%s' % cluster_template_id, data, 'cluster_template')