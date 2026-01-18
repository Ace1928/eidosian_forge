from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.lxd import (
from ansible.module_utils.basic import AnsibleModule
import os
def _apply_project_configs(self):
    """ Selection of the procedure: rebuild or merge

        The standard behavior is that all information not contained
        in the play is discarded.

        If "merge_project" is provides in the play and "True", then existing
        configurations from the project and new ones defined are merged.

        Args:
            None
        Kwargs:
            None
        Raises:
            None
        Returns:
            None"""
    old_config = dict()
    old_metadata = self.old_project_json['metadata'].copy()
    for attr in CONFIG_PARAMS:
        old_config[attr] = old_metadata[attr]
    if self.module.params['merge_project']:
        config = self._merge_dicts(self.config, old_config)
        if config == old_config:
            return
    else:
        config = self.config.copy()
    self.client.do('PUT', '/1.0/projects/{0}'.format(self.name), config)
    self.actions.append('apply_projects_configs')