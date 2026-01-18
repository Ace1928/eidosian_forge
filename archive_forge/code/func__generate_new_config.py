from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _generate_new_config(self, config):
    """ rebuild profile

        Rebuild the Profile by the configuration provided in the play.
        Existing configurations are discarded.

        This is the default behavior.

        Args:
            dict(config): Dict with the old config in 'metadata' and new config in 'config'
        Kwargs:
            None
        Raises:
            None
        Returns:
            dict(config): new config"""
    for k, v in self.config.items():
        config[k] = v
    return config