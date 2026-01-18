from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _namespace_has_migs(self, namespace, node=None):
    """returns a True or False.
        Does the namespace have migrations for the node passed?
        If no node passed, uses the local node or the first one in the list"""
    namespace_stats = self._info_cmd_helper('namespace/' + namespace, node)
    try:
        namespace_tx = int(namespace_stats[self.module.params['migrate_tx_key']])
        namespace_rx = int(namespace_stats[self.module.params['migrate_rx_key']])
    except KeyError:
        self.module.fail_json(msg='Did not find partition remaining key:' + self.module.params['migrate_tx_key'] + ' or key:' + self.module.params['migrate_rx_key'] + " in 'namespace/" + namespace + "' output.")
    except TypeError:
        self.module.fail_json(msg='namespace stat returned was not numerical')
    return namespace_tx != 0 or namespace_rx != 0