from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
class PodmanNetworkManager:
    """Module manager class.

    Defines according to parameters what actions should be applied to network
    """

    def __init__(self, module):
        """Initialize PodmanManager class.

        Arguments:
            module {obj} -- ansible module object
        """
        super(PodmanNetworkManager, self).__init__()
        self.module = module
        self.results = {'changed': False, 'actions': [], 'network': {}}
        self.name = self.module.params['name']
        self.executable = self.module.get_bin_path(self.module.params['executable'], required=True)
        self.state = self.module.params['state']
        self.recreate = self.module.params['recreate']
        self.network = PodmanNetwork(self.module, self.name)

    def update_network_result(self, changed=True):
        """Inspect the current network, update results with last info, exit.

        Keyword Arguments:
            changed {bool} -- whether any action was performed
                              (default: {True})
        """
        facts = self.network.get_info() if changed else self.network.info
        out, err = (self.network.stdout, self.network.stderr)
        self.results.update({'changed': changed, 'network': facts, 'podman_actions': self.network.actions}, stdout=out, stderr=err)
        if self.network.diff:
            self.results.update({'diff': self.network.diff})
        if self.module.params['debug']:
            self.results.update({'podman_version': self.network.version})
        self.module.exit_json(**self.results)

    def execute(self):
        """Execute the desired action according to map of actions & states."""
        states_map = {'present': self.make_present, 'absent': self.make_absent}
        process_action = states_map[self.state]
        process_action()
        self.module.fail_json(msg='Unexpected logic error happened, please contact maintainers ASAP!')

    def make_present(self):
        """Run actions if desired state is 'started'."""
        if not self.network.exists:
            self.network.create()
            self.results['actions'].append('created %s' % self.network.name)
            self.update_network_result()
        elif self.recreate or self.network.different:
            self.network.recreate()
            self.results['actions'].append('recreated %s' % self.network.name)
            self.update_network_result()
        else:
            self.update_network_result(changed=False)

    def make_absent(self):
        """Run actions if desired state is 'absent'."""
        if not self.network.exists:
            self.results.update({'changed': False})
        elif self.network.exists:
            self.network.delete()
            self.results['actions'].append('deleted %s' % self.network.name)
            self.results.update({'changed': True})
        self.results.update({'network': {}, 'podman_actions': self.network.actions})
        self.module.exit_json(**self.results)