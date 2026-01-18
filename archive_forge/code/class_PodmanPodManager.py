from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
class PodmanPodManager:
    """Module manager class.

    Defines according to parameters what actions should be applied to pod
    """

    def __init__(self, module, params):
        """Initialize PodmanManager class.

        Arguments:
            module {obj} -- ansible module object
        """
        self.module = module
        self.module_params = params
        self.results = {'changed': False, 'actions': [], 'pod': {}}
        self.name = self.module_params['name']
        self.executable = self.module.get_bin_path(self.module_params['executable'], required=True)
        self.state = self.module_params['state']
        self.recreate = self.module_params['recreate']
        self.pod = PodmanPod(self.module, self.name, self.module_params)

    def update_pod_result(self, changed=True):
        """Inspect the current pod, update results with last info, exit.

        Keyword Arguments:
            changed {bool} -- whether any action was performed
                              (default: {True})
        """
        facts = self.pod.get_info() if changed else self.pod.info
        out, err = (self.pod.stdout, self.pod.stderr)
        self.results.update({'changed': changed, 'pod': facts, 'podman_actions': self.pod.actions}, stdout=out, stderr=err)
        if self.pod.diff:
            self.results.update({'diff': self.pod.diff})
        if self.module.params['debug'] or self.module_params['debug']:
            self.results.update({'podman_version': self.pod.version})
        sysd = generate_systemd(self.module, self.module_params, self.name, self.pod.version)
        self.results['changed'] = changed or sysd['changed']
        self.results.update({'podman_systemd': sysd['systemd']})
        if sysd['diff']:
            if 'diff' not in self.results:
                self.results.update({'diff': sysd['diff']})
            else:
                self.results['diff']['before'] += sysd['diff']['before']
                self.results['diff']['after'] += sysd['diff']['after']

    def execute(self):
        """Execute the desired action according to map of actions & states."""
        states_map = {'created': self.make_created, 'started': self.make_started, 'stopped': self.make_stopped, 'restarted': self.make_restarted, 'absent': self.make_absent, 'killed': self.make_killed, 'paused': self.make_paused, 'unpaused': self.make_unpaused}
        process_action = states_map[self.state]
        process_action()
        return self.results

    def _create_or_recreate_pod(self):
        """Ensure pod exists and is exactly as it should be by input params."""
        changed = False
        if self.pod.exists:
            if self.pod.different or self.recreate:
                self.pod.recreate()
                self.results['actions'].append('recreated %s' % self.pod.name)
                changed = True
        elif not self.pod.exists:
            self.pod.create()
            self.results['actions'].append('created %s' % self.pod.name)
            changed = True
        return changed

    def make_created(self):
        """Run actions if desired state is 'created'."""
        if self.pod.exists and (not self.pod.different):
            self.update_pod_result(changed=False)
            return
        self._create_or_recreate_pod()
        self.update_pod_result()

    def make_killed(self):
        """Run actions if desired state is 'killed'."""
        self._create_or_recreate_pod()
        self.pod.kill()
        self.results['actions'].append('killed %s' % self.pod.name)
        self.update_pod_result()

    def make_paused(self):
        """Run actions if desired state is 'paused'."""
        changed = self._create_or_recreate_pod()
        if self.pod.paused:
            self.update_pod_result(changed=changed)
            return
        self.pod.pause()
        self.results['actions'].append('paused %s' % self.pod.name)
        self.update_pod_result()

    def make_unpaused(self):
        """Run actions if desired state is 'unpaused'."""
        changed = self._create_or_recreate_pod()
        if not self.pod.paused:
            self.update_pod_result(changed=changed)
            return
        self.pod.unpause()
        self.results['actions'].append('unpaused %s' % self.pod.name)
        self.update_pod_result()

    def make_started(self):
        """Run actions if desired state is 'started'."""
        changed = self._create_or_recreate_pod()
        if not changed and self.pod.running:
            self.update_pod_result(changed=changed)
            return
        self.pod.start()
        self.results['actions'].append('started %s' % self.pod.name)
        self.update_pod_result()

    def make_stopped(self):
        """Run actions if desired state is 'stopped'."""
        if not self.pod.exists:
            self.module.fail_json("Pod %s doesn't exist!" % self.pod.name)
        if self.pod.running:
            self.pod.stop()
            self.results['actions'].append('stopped %s' % self.pod.name)
            self.update_pod_result()
        elif self.pod.stopped:
            self.update_pod_result(changed=False)

    def make_restarted(self):
        """Run actions if desired state is 'restarted'."""
        if self.pod.exists:
            self.pod.restart()
            self.results['actions'].append('restarted %s' % self.pod.name)
            self.results.update({'changed': True})
            self.update_pod_result()
        else:
            self.module.fail_json("Pod %s doesn't exist!" % self.pod.name)

    def make_absent(self):
        """Run actions if desired state is 'absent'."""
        if not self.pod.exists:
            self.results.update({'changed': False})
        elif self.pod.exists:
            delete_systemd(self.module, self.module_params, self.name, self.pod.version)
            self.pod.delete()
            self.results['actions'].append('deleted %s' % self.pod.name)
            self.results.update({'changed': True})
        self.results.update({'pod': {}, 'podman_actions': self.pod.actions})