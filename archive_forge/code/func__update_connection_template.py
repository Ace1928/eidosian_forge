from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleResourceNotFound
def _update_connection_template(self, ethernet_network, bandwidth):
    if 'connectionTemplateUri' not in ethernet_network:
        return (False, None)
    connection_template = self.oneview_client.connection_templates.get(ethernet_network['connectionTemplateUri'])
    merged_data = connection_template.copy()
    merged_data.update({'bandwidth': bandwidth})
    if not self.compare(connection_template, merged_data):
        connection_template = self.oneview_client.connection_templates.update(merged_data)
        return (True, connection_template)
    else:
        return (False, None)