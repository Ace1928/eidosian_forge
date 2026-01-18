from __future__ import absolute_import, division, print_function
import json
import re
from ansible_collections.community.rabbitmq.plugins.module_utils.version import LooseVersion as Version
from ansible.module_utils.basic import AnsibleModule
class RabbitMqUserLimits(object):

    def __init__(self, module):
        self._module = module
        self._max_connections = module.params['max_connections']
        self._max_channels = module.params['max_channels']
        self._node = module.params['node']
        self._state = module.params['state']
        self._user = module.params['user']
        self._rabbitmqctl = module.get_bin_path('rabbitmqctl', True)
        self._version = self._rabbit_version()

    def _exec(self, args, force_exec_in_check_mode=False):
        if not self._module.check_mode or (self._module.check_mode and force_exec_in_check_mode):
            cmd = [self._rabbitmqctl, '-q']
            if self._node is not None:
                cmd.extend(['-n', self._node])
            rc, out, err = self._module.run_command(cmd + args, check_rc=True)
            return out
        return ''

    def _rabbit_version(self):
        status = self._exec(['status'], True)
        version_match = re.search('{rabbit,".*","(?P<version>.*)"}', status)
        if version_match:
            return Version(version_match.group('version'))
        version_match = re.search('RabbitMQ version: (?P<version>.*)', status)
        if version_match:
            return Version(version_match.group('version'))
        return None

    def _assert_version(self):
        if self._version and self._version < Version('3.8.10'):
            self._module.fail_json(changed=False, msg='User limits are only available for RabbitMQ >= 3.8.10. Detected version: %s' % self._version)

    def list(self):
        self._assert_version()
        exec_result = self._exec(['list_user_limits', '--user', self._user], False)
        max_connections = None
        max_channels = None
        if exec_result:
            user_limits = json.loads(exec_result)
            if 'max-connections' in user_limits:
                max_connections = user_limits['max-connections']
            if 'max-channels' in user_limits:
                max_channels = user_limits['max-channels']
        return dict(max_connections=max_connections, max_channels=max_channels)

    def set(self):
        self._assert_version()
        if self._module.check_mode:
            return
        if self._max_connections != -1:
            json_str = '{{"max-connections": {0}}}'.format(self._max_connections)
            self._exec(['set_user_limits', self._user, json_str])
        else:
            self._exec(['clear_user_limits', self._user, 'max-connections'])
        if self._max_channels != -1:
            json_str = '{{"max-channels": {0}}}'.format(self._max_channels)
            self._exec(['set_user_limits', self._user, json_str])
        else:
            self._exec(['clear_user_limits', self._user, 'max-channels'])

    def clear(self):
        self._assert_version()
        if self._module.check_mode:
            return
        return self._exec(['clear_user_limits', self._user, 'all'])