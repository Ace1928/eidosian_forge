from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
from ansible_collections.community.docker.plugins.module_utils.compose_v2 import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
class PullManager(BaseComposeManager):

    def __init__(self, client):
        super(PullManager, self).__init__(client)
        parameters = self.client.module.params
        self.policy = parameters['policy']
        if self.policy != 'always' and self.compose_version < LooseVersion('2.22.0'):
            self.client.fail('A pull policy other than always is only supported since Docker Compose 2.22.0. {0} has version {1}'.format(self.client.get_cli(), self.compose_version))

    def get_pull_cmd(self, dry_run, no_start=False):
        args = self.get_base_args() + ['pull']
        if self.policy != 'always':
            args.extend(['--policy', self.policy])
        if dry_run:
            args.append('--dry-run')
        args.append('--')
        return args

    def run(self):
        result = dict()
        args = self.get_pull_cmd(self.check_mode)
        rc, stdout, stderr = self.client.call_cli(*args, cwd=self.project_src)
        events = self.parse_events(stderr, dry_run=self.check_mode)
        self.emit_warnings(events)
        self.update_result(result, events, stdout, stderr, ignore_service_pull_events=self.policy != 'missing' and (not self.check_mode))
        self.update_failed(result, events, args, stdout, stderr, rc)
        self.cleanup_result(result)
        return result