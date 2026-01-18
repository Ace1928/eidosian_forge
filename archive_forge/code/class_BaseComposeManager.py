from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
class BaseComposeManager(DockerBaseClass):

    def __init__(self, client, min_version=MINIMUM_COMPOSE_VERSION):
        super(BaseComposeManager, self).__init__()
        self.client = client
        self.check_mode = self.client.check_mode
        parameters = self.client.module.params
        self.project_src = parameters['project_src']
        self.project_name = parameters['project_name']
        self.files = parameters['files']
        self.env_files = parameters['env_files']
        self.profiles = parameters['profiles']
        compose = self.client.get_client_plugin_info('compose')
        if compose is None:
            self.client.fail('Docker CLI {0} does not have the compose plugin installed'.format(self.client.get_cli()))
        compose_version = compose['Version'].lstrip('v')
        self.compose_version = LooseVersion(compose_version)
        if self.compose_version < LooseVersion(min_version):
            self.client.fail('Docker CLI {cli} has the compose plugin with version {version}; need version {min_version} or later'.format(cli=self.client.get_cli(), version=compose_version, min_version=min_version))
        if not os.path.isdir(self.project_src):
            self.client.fail('"{0}" is not a directory'.format(self.project_src))
        if self.files:
            for file in self.files:
                path = os.path.join(self.project_src, file)
                if not os.path.exists(path):
                    self.client.fail('Cannot find Compose file "{0}" relative to project directory "{1}"'.format(file, self.project_src))
        elif all((not os.path.exists(os.path.join(self.project_src, f)) for f in DOCKER_COMPOSE_FILES)):
            filenames = ', '.join(DOCKER_COMPOSE_FILES[:-1])
            self.client.fail('"{0}" does not contain {1}, or {2}'.format(self.project_src, filenames, DOCKER_COMPOSE_FILES[-1]))

    def get_base_args(self):
        args = ['compose', '--ansi', 'never']
        if self.compose_version >= LooseVersion('2.19.0'):
            args.extend(['--progress', 'plain'])
        args.extend(['--project-directory', self.project_src])
        if self.project_name:
            args.extend(['--project-name', self.project_name])
        for file in self.files or []:
            args.extend(['--file', file])
        for env_file in self.env_files or []:
            args.extend(['--env-file', env_file])
        for profile in self.profiles or []:
            args.extend(['--profile', profile])
        return args

    def list_containers_raw(self):
        args = self.get_base_args() + ['ps', '--format', 'json', '--all']
        if self.compose_version >= LooseVersion('2.23.0'):
            args.append('--no-trunc')
        kwargs = dict(cwd=self.project_src, check_rc=True)
        if self.compose_version >= LooseVersion('2.21.0'):
            dummy, containers, dummy = self.client.call_cli_json_stream(*args, **kwargs)
        else:
            dummy, containers, dummy = self.client.call_cli_json(*args, **kwargs)
        return containers

    def list_containers(self):
        result = []
        for container in self.list_containers_raw():
            labels = {}
            if container.get('Labels'):
                for part in container['Labels'].split(','):
                    label_value = part.split('=', 1)
                    labels[label_value[0]] = label_value[1] if len(label_value) > 1 else ''
            container['Labels'] = labels
            container['Names'] = container.get('Names', container['Name']).split(',')
            container['Networks'] = container.get('Networks', '').split(',')
            container['Publishers'] = container.get('Publishers') or []
            result.append(container)
        return result

    def list_images(self):
        args = self.get_base_args() + ['images', '--format', 'json']
        kwargs = dict(cwd=self.project_src, check_rc=True)
        dummy, images, dummy = self.client.call_cli_json(*args, **kwargs)
        return images

    def parse_events(self, stderr, dry_run=False):
        return parse_events(stderr, dry_run=dry_run, warn_function=self.client.warn)

    def emit_warnings(self, events):
        emit_warnings(events, warn_function=self.client.warn)

    def update_result(self, result, events, stdout, stderr, ignore_service_pull_events=False):
        result['changed'] = result.get('changed', False) or has_changes(events, ignore_service_pull_events=ignore_service_pull_events)
        result['actions'] = result.get('actions', []) + extract_actions(events)
        result['stdout'] = combine_text_output(result.get('stdout'), to_native(stdout))
        result['stderr'] = combine_text_output(result.get('stderr'), to_native(stderr))

    def update_failed(self, result, events, args, stdout, stderr, rc):
        return update_failed(result, events, args=args, stdout=stdout, stderr=stderr, rc=rc, cli=self.client.get_cli())

    def cleanup_result(self, result):
        if not result.get('failed'):
            for res in ('stdout', 'stderr'):
                if result.get(res) == '':
                    result.pop(res)