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
class PodmanContainer:
    """Perform container tasks.

    Manages podman container, inspects it and checks its current state
    """

    def __init__(self, module, name, module_params):
        """Initialize PodmanContainer class.

        Arguments:
            module {obj} -- ansible module object
            name {str} -- name of container
        """
        self.module = module
        self.module_params = module_params
        self.name = name
        self.stdout, self.stderr = ('', '')
        self.info = self.get_info()
        self.version = self._get_podman_version()
        self.diff = {}
        self.actions = []

    @property
    def exists(self):
        """Check if container exists."""
        return bool(self.info != {})

    @property
    def different(self):
        """Check if container is different."""
        diffcheck = PodmanContainerDiff(self.module, self.module_params, self.info, self.get_image_info(), self.version)
        is_different = diffcheck.is_different()
        diffs = diffcheck.diff
        if self.module._diff and is_different and diffs['before'] and diffs['after']:
            self.diff['before'] = '\n'.join(['%s - %s' % (k, v) for k, v in sorted(diffs['before'].items())]) + '\n'
            self.diff['after'] = '\n'.join(['%s - %s' % (k, v) for k, v in sorted(diffs['after'].items())]) + '\n'
        return is_different

    @property
    def running(self):
        """Return True if container is running now."""
        return self.exists and self.info['State']['Running']

    @property
    def stopped(self):
        """Return True if container exists and is not running now."""
        return self.exists and (not self.info['State']['Running'])

    def get_info(self):
        """Inspect container and gather info about it."""
        rc, out, err = self.module.run_command([self.module_params['executable'], b'container', b'inspect', self.name])
        return json.loads(out)[0] if rc == 0 else {}

    def get_image_info(self):
        """Inspect container image and gather info about it."""
        is_rootfs = self.module_params['rootfs']
        if is_rootfs:
            return {'Id': self.module_params['image']}
        rc, out, err = self.module.run_command([self.module_params['executable'], b'image', b'inspect', self.module_params['image']])
        return json.loads(out)[0] if rc == 0 else {}

    def _get_podman_version(self):
        rc, out, err = self.module.run_command([self.module_params['executable'], b'--version'])
        if rc != 0 or not out or 'version' not in out:
            self.module.fail_json(msg='%s run failed!' % self.module_params['executable'])
        return out.split('version')[1].strip()

    def _perform_action(self, action):
        """Perform action with container.

        Arguments:
            action {str} -- action to perform - start, create, stop, run,
                            delete, restart
        """
        b_command = PodmanModuleParams(action, self.module_params, self.version, self.module).construct_command_from_params()
        full_cmd = ' '.join([self.module_params['executable']] + [to_native(i) for i in b_command])
        self.actions.append(full_cmd)
        if self.module.check_mode:
            self.module.log('PODMAN-CONTAINER-DEBUG (check_mode): %s' % full_cmd)
        else:
            rc, out, err = self.module.run_command([self.module_params['executable'], b'container'] + b_command, expand_user_and_vars=False)
            self.module.log('PODMAN-CONTAINER-DEBUG: %s' % full_cmd)
            if self.module_params['debug']:
                self.module.log('PODMAN-CONTAINER-DEBUG STDOUT: %s' % out)
                self.module.log('PODMAN-CONTAINER-DEBUG STDERR: %s' % err)
                self.module.log('PODMAN-CONTAINER-DEBUG RC: %s' % rc)
            self.stdout = out
            self.stderr = err
            if rc != 0:
                self.module.fail_json(msg='Container %s exited with code %s when %sed' % (self.name, rc, action), stdout=out, stderr=err)

    def run(self):
        """Run the container."""
        self._perform_action('run')

    def delete(self):
        """Delete the container."""
        self._perform_action('delete')

    def stop(self):
        """Stop the container."""
        self._perform_action('stop')

    def start(self):
        """Start the container."""
        self._perform_action('start')

    def restart(self):
        """Restart the container."""
        self._perform_action('restart')

    def create(self):
        """Create the container."""
        self._perform_action('create')

    def recreate(self):
        """Recreate the container."""
        if self.running:
            self.stop()
        if not self.info['HostConfig']['AutoRemove']:
            self.delete()
        self.create()

    def recreate_run(self):
        """Recreate and run the container."""
        if self.running:
            self.stop()
        if not self.info['HostConfig']['AutoRemove']:
            self.delete()
        self.run()