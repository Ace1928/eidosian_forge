from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
class PodmanPod:
    """Perform pod tasks.

    Manages podman pod, inspects it and checks its current state
    """

    def __init__(self, module, name, module_params):
        """Initialize PodmanPod class.

        Arguments:
            module {obj} -- ansible module object
            name {str} -- name of pod
        """
        self.module = module
        self.module_params = module_params
        self.name = name
        self.stdout, self.stderr = ('', '')
        self.info = self.get_info()
        self.infra_info = self.get_infra_info()
        self.version = self._get_podman_version()
        self.diff = {}
        self.actions = []

    @property
    def exists(self):
        """Check if pod exists."""
        return bool(self.info != {})

    @property
    def different(self):
        """Check if pod is different."""
        diffcheck = PodmanPodDiff(self.module, self.module_params, self.info, self.infra_info, self.version)
        is_different = diffcheck.is_different()
        diffs = diffcheck.diff
        if self.module._diff and is_different and diffs['before'] and diffs['after']:
            self.diff['before'] = '\n'.join(['%s - %s' % (k, v) for k, v in sorted(diffs['before'].items())]) + '\n'
            self.diff['after'] = '\n'.join(['%s - %s' % (k, v) for k, v in sorted(diffs['after'].items())]) + '\n'
        return is_different

    @property
    def running(self):
        """Return True if pod is running now."""
        if 'status' in self.info['State']:
            return self.info['State']['status'] == 'Running'
        ps_info = self.get_ps()
        if 'status' in ps_info:
            return ps_info['status'] == 'Running'
        return self.info['State'] == 'Running'

    @property
    def paused(self):
        """Return True if pod is paused now."""
        if 'status' in self.info['State']:
            return self.info['State']['status'] == 'Paused'
        return self.info['State'] == 'Paused'

    @property
    def stopped(self):
        """Return True if pod exists and is not running now."""
        if not self.exists:
            return False
        if 'status' in self.info['State']:
            return not self.info['State']['status'] == 'Running'
        return not self.info['State'] == 'Running'

    def get_info(self):
        """Inspect pod and gather info about it."""
        rc, out, err = self.module.run_command([self.module_params['executable'], b'pod', b'inspect', self.name])
        return json.loads(out) if rc == 0 else {}

    def get_ps(self):
        """Inspect pod process and gather info about it."""
        rc, out, err = self.module.run_command([self.module_params['executable'], b'pod', b'ps', b'--format', b'json', b'--filter', 'name=' + self.name])
        return json.loads(out)[0] if rc == 0 else {}

    def get_infra_info(self):
        """Inspect pod and gather info about it."""
        if not self.info:
            return {}
        if 'InfraContainerID' in self.info:
            infra_container_id = self.info['InfraContainerID']
        elif 'State' in self.info and 'infraContainerID' in self.info['State']:
            infra_container_id = self.info['State']['infraContainerID']
        else:
            return {}
        rc, out, err = self.module.run_command([self.module_params['executable'], b'inspect', infra_container_id])
        return json.loads(out)[0] if rc == 0 else {}

    def _get_podman_version(self):
        rc, out, err = self.module.run_command([self.module_params['executable'], b'--version'])
        if rc != 0 or not out or 'version' not in out:
            self.module.fail_json(msg='%s run failed!' % self.module_params['executable'])
        return out.split('version')[1].strip()

    def _perform_action(self, action):
        """Perform action with pod.

        Arguments:
            action {str} -- action to perform - start, create, stop, pause
                            unpause, delete, restart, kill
        """
        b_command = PodmanPodModuleParams(action, self.module_params, self.version, self.module).construct_command_from_params()
        full_cmd = ' '.join([self.module_params['executable'], 'pod'] + [to_native(i) for i in b_command])
        self.module.log('PODMAN-POD-DEBUG: %s' % full_cmd)
        self.actions.append(full_cmd)
        if not self.module.check_mode:
            rc, out, err = self.module.run_command([self.module_params['executable'], b'pod'] + b_command, expand_user_and_vars=False)
            self.stdout = out
            self.stderr = err
            if rc != 0:
                self.module.fail_json(msg="Can't %s pod %s" % (action, self.name), stdout=out, stderr=err)

    def delete(self):
        """Delete the pod."""
        self._perform_action('delete')

    def stop(self):
        """Stop the pod."""
        self._perform_action('stop')

    def start(self):
        """Start the pod."""
        self._perform_action('start')

    def create(self):
        """Create the pod."""
        self._perform_action('create')

    def recreate(self):
        """Recreate the pod."""
        self.delete()
        self.create()

    def restart(self):
        """Restart the pod."""
        self._perform_action('restart')

    def kill(self):
        """Kill the pod."""
        self._perform_action('kill')

    def pause(self):
        """Pause the pod."""
        self._perform_action('pause')

    def unpause(self):
        """Unpause the pod."""
        self._perform_action('unpause')