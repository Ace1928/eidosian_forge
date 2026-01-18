from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _apply_profile_configs(self):
    """ Selection of the procedure: rebuild or merge

        The standard behavior is that all information not contained
        in the play is discarded.

        If "merge_profile" is provides in the play and "True", then existing
        configurations from the profile and new ones defined are merged.

        Args:
            None
        Kwargs:
            None
        Raises:
            None
        Returns:
            None"""
    config = self.old_profile_json.copy()
    if self.module.params['merge_profile']:
        config = self._merge_config(config)
    else:
        config = self._generate_new_config(config)
    url = '/1.0/profiles/{0}'.format(self.name)
    if self.project:
        url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
    self.client.do('PUT', url, config)
    self.actions.append('apply_profile_configs')