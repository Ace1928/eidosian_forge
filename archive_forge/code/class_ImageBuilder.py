from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
class ImageBuilder(DockerBaseClass):

    def __init__(self, client):
        super(ImageBuilder, self).__init__()
        self.client = client
        self.check_mode = self.client.check_mode
        parameters = self.client.module.params
        self.cache_from = parameters['cache_from']
        self.pull = parameters['pull']
        self.network = parameters['network']
        self.nocache = parameters['nocache']
        self.etc_hosts = clean_dict_booleans_for_docker_api(parameters['etc_hosts'])
        self.args = clean_dict_booleans_for_docker_api(parameters['args'])
        self.target = parameters['target']
        self.platform = parameters['platform']
        self.shm_size = convert_to_bytes(parameters['shm_size'], self.client.module, 'shm_size')
        self.labels = clean_dict_booleans_for_docker_api(parameters['labels'])
        self.rebuild = parameters['rebuild']
        buildx = self.client.get_client_plugin_info('buildx')
        if buildx is None:
            self.fail('Docker CLI {0} does not have the buildx plugin installed'.format(self.client.get_cli()))
        self.path = parameters['path']
        if not os.path.isdir(self.path):
            self.fail('"{0}" is not an existing directory'.format(self.path))
        self.dockerfile = parameters['dockerfile']
        if self.dockerfile and (not os.path.isfile(os.path.join(self.path, self.dockerfile))):
            self.fail('"{0}" is not an existing file'.format(os.path.join(self.path, self.dockerfile)))
        self.name = parameters['name']
        self.tag = parameters['tag']
        if not is_valid_tag(self.tag, allow_empty=True):
            self.fail('"{0}" is not a valid docker tag'.format(self.tag))
        if is_image_name_id(self.name):
            self.fail('Image name must not be a digest')
        repo, repo_tag = parse_repository_tag(self.name)
        if repo_tag:
            self.name = repo
            self.tag = repo_tag
        if is_image_name_id(self.tag):
            self.fail('Image name must not contain a digest, but have a tag')

    def fail(self, msg, **kwargs):
        self.client.fail(msg, **kwargs)

    def add_list_arg(self, args, option, values):
        for value in values:
            args.extend([option, value])

    def add_args(self, args):
        args.extend(['--tag', '%s:%s' % (self.name, self.tag)])
        if self.dockerfile:
            args.extend(['--file', os.path.join(self.path, self.dockerfile)])
        if self.cache_from:
            self.add_list_arg(args, '--cache-from', self.cache_from)
        if self.pull:
            args.append('--pull')
        if self.network:
            args.extend(['--network', self.network])
        if self.nocache:
            args.append('--no-cache')
        if self.etc_hosts:
            self.add_list_arg(args, '--add-host', dict_to_list(self.etc_hosts, ':'))
        if self.args:
            self.add_list_arg(args, '--build-arg', dict_to_list(self.args))
        if self.target:
            args.extend(['--target', self.target])
        if self.platform:
            args.extend(['--platform', self.platform])
        if self.shm_size:
            args.extend(['--shm-size', str(self.shm_size)])
        if self.labels:
            self.add_list_arg(args, '--label', dict_to_list(self.labels))

    def build_image(self):
        image = self.client.find_image(self.name, self.tag)
        results = dict(changed=False, actions=[], image=image or {})
        if image:
            if self.rebuild == 'never':
                return results
        results['changed'] = True
        if not self.check_mode:
            args = ['buildx', 'build', '--progress', 'plain']
            self.add_args(args)
            args.extend(['--', self.path])
            rc, stdout, stderr = self.client.call_cli(*args)
            if rc != 0:
                self.fail('Building %s:%s failed' % (self.name, self.tag), stdout=to_native(stdout), stderr=to_native(stderr))
            results['stdout'] = to_native(stdout)
            results['stderr'] = to_native(stderr)
            results['image'] = self.client.find_image(self.name, self.tag) or {}
        return results