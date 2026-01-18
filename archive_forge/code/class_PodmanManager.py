from __future__ import (absolute_import, division, print_function)
import json  # noqa: F402
import os  # noqa: F402
import shlex  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import normalize_signal
from ansible_collections.containers.podman.plugins.module_utils.podman.common import ARGUMENTS_OPTS_DICT
class PodmanManager:
    """Module manager class.

    Defines according to parameters what actions should be applied to container
    """

    def __init__(self, module, params):
        """Initialize PodmanManager class.

        Arguments:
            module {obj} -- ansible module object
        """
        self.module = module
        self.results = {'changed': False, 'actions': [], 'container': {}}
        self.module_params = params
        self.name = self.module_params['name']
        self.executable = self.module.get_bin_path(self.module_params['executable'], required=True)
        self.image = self.module_params['image']
        image_actions = ensure_image_exists(self.module, self.image, self.module_params)
        self.results['actions'] += image_actions
        self.state = self.module_params['state']
        self.restart = self.module_params['force_restart']
        self.recreate = self.module_params['recreate']
        if self.module_params['generate_systemd'].get('new'):
            self.module_params['rm'] = True
        self.container = PodmanContainer(self.module, self.name, self.module_params)

    def update_container_result(self, changed=True):
        """Inspect the current container, update results with last info, exit.

        Keyword Arguments:
            changed {bool} -- whether any action was performed
                              (default: {True})
        """
        facts = self.container.get_info() if changed else self.container.info
        out, err = (self.container.stdout, self.container.stderr)
        self.results.update({'changed': changed, 'container': facts, 'podman_actions': self.container.actions}, stdout=out, stderr=err)
        if self.container.diff:
            self.results.update({'diff': self.container.diff})
        if self.module.params['debug'] or self.module_params['debug']:
            self.results.update({'podman_version': self.container.version})
        sysd = generate_systemd(self.module, self.module_params, self.name, self.container.version)
        self.results['changed'] = changed or sysd['changed']
        self.results.update({'podman_systemd': sysd['systemd']})
        if sysd['diff']:
            if 'diff' not in self.results:
                self.results.update({'diff': sysd['diff']})
            else:
                self.results['diff']['before'] += sysd['diff']['before']
                self.results['diff']['after'] += sysd['diff']['after']

    def make_started(self):
        """Run actions if desired state is 'started'."""
        if not self.image:
            if not self.container.exists:
                self.module.fail_json(msg='Cannot start container when image is not specified!')
            if self.restart:
                self.container.restart()
                self.results['actions'].append('restarted %s' % self.container.name)
            else:
                self.container.start()
                self.results['actions'].append('started %s' % self.container.name)
            self.update_container_result()
            return
        if self.container.exists and self.restart:
            if self.container.running:
                self.container.restart()
                self.results['actions'].append('restarted %s' % self.container.name)
            else:
                self.container.start()
                self.results['actions'].append('started %s' % self.container.name)
            self.update_container_result()
            return
        if self.container.running and (self.container.different or self.recreate):
            self.container.recreate_run()
            self.results['actions'].append('recreated %s' % self.container.name)
            self.update_container_result()
            return
        elif self.container.running and (not self.container.different):
            if self.restart:
                self.container.restart()
                self.results['actions'].append('restarted %s' % self.container.name)
                self.update_container_result()
                return
            self.update_container_result(changed=False)
            return
        elif not self.container.exists:
            self.container.run()
            self.results['actions'].append('started %s' % self.container.name)
            self.update_container_result()
            return
        elif self.container.stopped and (self.container.different or self.recreate):
            self.container.recreate_run()
            self.results['actions'].append('recreated %s' % self.container.name)
            self.update_container_result()
            return
        elif self.container.stopped and (not self.container.different):
            self.container.start()
            self.results['actions'].append('started %s' % self.container.name)
            self.update_container_result()
            return

    def make_created(self):
        """Run actions if desired state is 'created'."""
        if not self.container.exists and (not self.image):
            self.module.fail_json(msg='Cannot create container when image is not specified!')
        if not self.container.exists:
            self.container.create()
            self.results['actions'].append('created %s' % self.container.name)
            self.update_container_result()
            return
        else:
            if self.container.different or self.recreate:
                self.container.recreate()
                self.results['actions'].append('recreated %s' % self.container.name)
                if self.container.running:
                    self.container.start()
                    self.results['actions'].append('started %s' % self.container.name)
                self.update_container_result()
                return
            elif self.restart:
                if self.container.running:
                    self.container.restart()
                    self.results['actions'].append('restarted %s' % self.container.name)
                else:
                    self.container.start()
                    self.results['actions'].append('started %s' % self.container.name)
                self.update_container_result()
                return
            self.update_container_result(changed=False)
            return

    def make_stopped(self):
        """Run actions if desired state is 'stopped'."""
        if not self.container.exists and (not self.image):
            self.module.fail_json(msg='Cannot create container when image is not specified!')
        if not self.container.exists:
            self.container.create()
            self.results['actions'].append('created %s' % self.container.name)
            self.update_container_result()
            return
        if self.container.stopped:
            self.update_container_result(changed=False)
            return
        elif self.container.running:
            self.container.stop()
            self.results['actions'].append('stopped %s' % self.container.name)
            self.update_container_result()
            return

    def make_absent(self):
        """Run actions if desired state is 'absent'."""
        if not self.container.exists:
            self.results.update({'changed': False})
        elif self.container.exists:
            delete_systemd(self.module, self.module_params, self.name, self.container.version)
            self.container.delete()
            self.results['actions'].append('deleted %s' % self.container.name)
            self.results.update({'changed': True})
        self.results.update({'container': {}, 'podman_actions': self.container.actions})

    def execute(self):
        """Execute the desired action according to map of actions & states."""
        states_map = {'present': self.make_created, 'started': self.make_started, 'absent': self.make_absent, 'stopped': self.make_stopped, 'created': self.make_created}
        process_action = states_map[self.state]
        process_action()
        return self.results