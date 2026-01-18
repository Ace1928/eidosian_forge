from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.validation import check_type_int
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
from ansible_collections.community.docker.plugins.module_utils.compose_v2 import (
class ServicesManager(BaseComposeManager):

    def __init__(self, client):
        super(ServicesManager, self).__init__(client)
        parameters = self.client.module.params
        self.state = parameters['state']
        self.dependencies = parameters['dependencies']
        self.pull = parameters['pull']
        self.build = parameters['build']
        self.recreate = parameters['recreate']
        self.remove_images = parameters['remove_images']
        self.remove_volumes = parameters['remove_volumes']
        self.remove_orphans = parameters['remove_orphans']
        self.timeout = parameters['timeout']
        self.services = parameters['services'] or []
        self.scale = parameters['scale'] or {}
        self.wait = parameters['wait']
        self.wait_timeout = parameters['wait_timeout']
        for key, value in self.scale.items():
            if not isinstance(key, string_types):
                self.client.fail('The key %s for `scale` is not a string' % repr(key))
            try:
                value = check_type_int(value)
            except TypeError as exc:
                self.client.fail('The value %s for `scale[%s]` is not an integer' % (repr(value), repr(key)))
            if value < 0:
                self.client.fail('The value %s for `scale[%s]` is negative' % (repr(value), repr(key)))
            self.scale[key] = value

    def run(self):
        if self.state == 'present':
            result = self.cmd_up()
        elif self.state == 'stopped':
            result = self.cmd_stop()
        elif self.state == 'restarted':
            result = self.cmd_restart()
        elif self.state == 'absent':
            result = self.cmd_down()
        result['containers'] = self.list_containers()
        result['images'] = self.list_images()
        self.cleanup_result(result)
        return result

    def get_up_cmd(self, dry_run, no_start=False):
        args = self.get_base_args() + ['up', '--detach', '--no-color', '--quiet-pull']
        if self.pull != 'policy':
            args.extend(['--pull', self.pull])
        if self.remove_orphans:
            args.append('--remove-orphans')
        if self.recreate == 'always':
            args.append('--force-recreate')
        if self.recreate == 'never':
            args.append('--no-recreate')
        if not self.dependencies:
            args.append('--no-deps')
        if self.timeout is not None:
            args.extend(['--timeout', '%d' % self.timeout])
        if self.build == 'always':
            args.append('--build')
        elif self.build == 'never':
            args.append('--no-build')
        for key, value in sorted(self.scale.items()):
            args.extend(['--scale', '%s=%d' % (key, value)])
        if self.wait:
            args.append('--wait')
            if self.wait_timeout is not None:
                args.extend(['--wait-timeout', str(self.wait_timeout)])
        if no_start:
            args.append('--no-start')
        if dry_run:
            args.append('--dry-run')
        for service in self.services:
            args.append(service)
        args.append('--')
        return args

    def cmd_up(self):
        result = dict()
        args = self.get_up_cmd(self.check_mode)
        rc, stdout, stderr = self.client.call_cli(*args, cwd=self.project_src)
        events = self.parse_events(stderr, dry_run=self.check_mode)
        self.emit_warnings(events)
        self.update_result(result, events, stdout, stderr, ignore_service_pull_events=True)
        self.update_failed(result, events, args, stdout, stderr, rc)
        return result

    def get_stop_cmd(self, dry_run):
        args = self.get_base_args() + ['stop']
        if self.timeout is not None:
            args.extend(['--timeout', '%d' % self.timeout])
        if dry_run:
            args.append('--dry-run')
        for service in self.services:
            args.append(service)
        args.append('--')
        return args

    def _are_containers_stopped(self):
        for container in self.list_containers_raw():
            if container['State'] not in ('created', 'exited', 'stopped', 'killed'):
                return False
        return True

    def cmd_stop(self):
        result = dict()
        args_1 = self.get_up_cmd(self.check_mode, no_start=True)
        rc_1, stdout_1, stderr_1 = self.client.call_cli(*args_1, cwd=self.project_src)
        events_1 = self.parse_events(stderr_1, dry_run=self.check_mode)
        self.emit_warnings(events_1)
        self.update_result(result, events_1, stdout_1, stderr_1, ignore_service_pull_events=True)
        is_failed_1 = is_failed(events_1, rc_1)
        if not is_failed_1 and (not self._are_containers_stopped()):
            args_2 = self.get_stop_cmd(self.check_mode)
            rc_2, stdout_2, stderr_2 = self.client.call_cli(*args_2, cwd=self.project_src)
            events_2 = self.parse_events(stderr_2, dry_run=self.check_mode)
            self.emit_warnings(events_2)
            self.update_result(result, events_2, stdout_2, stderr_2)
        else:
            args_2 = []
            rc_2, stdout_2, stderr_2 = (0, b'', b'')
            events_2 = []
        self.update_failed(result, events_1 + events_2, args_1 if is_failed_1 else args_2, stdout_1 if is_failed_1 else stdout_2, stderr_1 if is_failed_1 else stderr_2, rc_1 if is_failed_1 else rc_2)
        return result

    def get_restart_cmd(self, dry_run):
        args = self.get_base_args() + ['restart']
        if not self.dependencies:
            args.append('--no-deps')
        if self.timeout is not None:
            args.extend(['--timeout', '%d' % self.timeout])
        if dry_run:
            args.append('--dry-run')
        for service in self.services:
            args.append(service)
        args.append('--')
        return args

    def cmd_restart(self):
        result = dict()
        args = self.get_restart_cmd(self.check_mode)
        rc, stdout, stderr = self.client.call_cli(*args, cwd=self.project_src)
        events = self.parse_events(stderr, dry_run=self.check_mode)
        self.emit_warnings(events)
        self.update_result(result, events, stdout, stderr)
        self.update_failed(result, events, args, stdout, stderr, rc)
        return result

    def get_down_cmd(self, dry_run):
        args = self.get_base_args() + ['down']
        if self.remove_orphans:
            args.append('--remove-orphans')
        if self.remove_images:
            args.extend(['--rmi', self.remove_images])
        if self.remove_volumes:
            args.append('--volumes')
        if self.timeout is not None:
            args.extend(['--timeout', '%d' % self.timeout])
        if dry_run:
            args.append('--dry-run')
        for service in self.services:
            args.append(service)
        args.append('--')
        return args

    def cmd_down(self):
        result = dict()
        args = self.get_down_cmd(self.check_mode)
        rc, stdout, stderr = self.client.call_cli(*args, cwd=self.project_src)
        events = self.parse_events(stderr, dry_run=self.check_mode)
        self.emit_warnings(events)
        self.update_result(result, events, stdout, stderr)
        self.update_failed(result, events, args, stdout, stderr, rc)
        return result