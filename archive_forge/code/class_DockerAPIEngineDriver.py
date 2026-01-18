from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils.module_container.base import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.errors import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
class DockerAPIEngineDriver(EngineDriver):
    name = 'docker_api'

    def setup(self, argument_spec, mutually_exclusive=None, required_together=None, required_one_of=None, required_if=None, required_by=None):
        argument_spec = argument_spec or {}
        mutually_exclusive = mutually_exclusive or []
        required_together = required_together or []
        required_one_of = required_one_of or []
        required_if = required_if or []
        required_by = required_by or {}
        active_options = []
        option_minimal_versions = {}
        for options in OPTIONS:
            if not options.supports_engine(self.name):
                continue
            mutually_exclusive.extend(options.ansible_mutually_exclusive)
            required_together.extend(options.ansible_required_together)
            required_one_of.extend(options.ansible_required_one_of)
            required_if.extend(options.ansible_required_if)
            required_by.update(options.ansible_required_by)
            argument_spec.update(options.argument_spec)
            engine = options.get_engine(self.name)
            if engine.min_api_version is not None:
                for option in options.options:
                    if not option.not_an_ansible_option:
                        option_minimal_versions[option.name] = {'docker_api_version': engine.min_api_version}
            if engine.extra_option_minimal_versions:
                option_minimal_versions.update(engine.extra_option_minimal_versions)
            active_options.append(options)
        client = AnsibleDockerClient(argument_spec=argument_spec, mutually_exclusive=mutually_exclusive, required_together=required_together, required_one_of=required_one_of, required_if=required_if, required_by=required_by, option_minimal_versions=option_minimal_versions, supports_check_mode=True)
        return (client.module, active_options, client)

    def get_host_info(self, client):
        return client.info()

    def get_api_version(self, client):
        return client.docker_api_version

    def get_container_id(self, container):
        return container['Id']

    def get_image_from_container(self, container):
        return container['Image']

    def get_image_name_from_container(self, container):
        return container['Config'].get('Image')

    def is_container_removing(self, container):
        if container.get('State'):
            return container['State'].get('Status') == 'removing'
        return False

    def is_container_running(self, container):
        if container.get('State'):
            if container['State'].get('Running') and (not container['State'].get('Ghost', False)):
                return True
        return False

    def is_container_paused(self, container):
        if container.get('State'):
            return container['State'].get('Paused', False)
        return False

    def inspect_container_by_name(self, client, container_name):
        return client.get_container(container_name)

    def inspect_container_by_id(self, client, container_id):
        return client.get_container_by_id(container_id)

    def inspect_image_by_id(self, client, image_id):
        return client.find_image_by_id(image_id, accept_missing_image=True)

    def inspect_image_by_name(self, client, repository, tag):
        return client.find_image(repository, tag)

    def pull_image(self, client, repository, tag, platform=None):
        return client.pull_image(repository, tag, platform=platform)

    def pause_container(self, client, container_id):
        client.post_call('/containers/{0}/pause', container_id)

    def unpause_container(self, client, container_id):
        client.post_call('/containers/{0}/unpause', container_id)

    def disconnect_container_from_network(self, client, container_id, network_id):
        client.post_json('/networks/{0}/disconnect', network_id, data={'Container': container_id})

    def connect_container_to_network(self, client, container_id, network_id, parameters=None):
        parameters = (parameters or {}).copy()
        params = {}
        for para, dest_para in {'ipv4_address': 'IPv4Address', 'ipv6_address': 'IPv6Address', 'links': 'Links', 'aliases': 'Aliases', 'mac_address': 'MacAddress'}.items():
            value = parameters.pop(para, None)
            if value:
                if para == 'links':
                    value = normalize_links(value)
                params[dest_para] = value
        if parameters:
            raise Exception('Unknown parameter(s) for connect_container_to_network for Docker API driver: %s' % ', '.join(['"%s"' % p for p in sorted(parameters)]))
        ipam_config = {}
        for param in ('IPv4Address', 'IPv6Address'):
            if param in params:
                ipam_config[param] = params.pop(param)
        if ipam_config:
            params['IPAMConfig'] = ipam_config
        data = {'Container': container_id, 'EndpointConfig': params}
        client.post_json('/networks/{0}/connect', network_id, data=data)

    def create_container(self, client, container_name, create_parameters):
        params = {'name': container_name}
        if 'platform' in create_parameters:
            params['platform'] = create_parameters.pop('platform')
        new_container = client.post_json_to_json('/containers/create', data=create_parameters, params=params)
        client.report_warnings(new_container)
        return new_container['Id']

    def start_container(self, client, container_id):
        client.post_json('/containers/{0}/start', container_id)

    def wait_for_container(self, client, container_id, timeout=None):
        return client.post_json_to_json('/containers/{0}/wait', container_id, timeout=timeout)['StatusCode']

    def get_container_output(self, client, container_id):
        config = client.get_json('/containers/{0}/json', container_id)
        logging_driver = config['HostConfig']['LogConfig']['Type']
        if logging_driver in ('json-file', 'journald', 'local'):
            params = {'stderr': 1, 'stdout': 1, 'timestamps': 0, 'follow': 0, 'tail': 'all'}
            res = client._get(client._url('/containers/{0}/logs', container_id), params=params)
            output = client._get_result_tty(False, res, config['Config']['Tty'])
            return (output, True)
        else:
            return ('Result logged using `%s` driver' % logging_driver, False)

    def update_container(self, client, container_id, update_parameters):
        result = client.post_json_to_json('/containers/{0}/update', container_id, data=update_parameters)
        client.report_warnings(result)

    def restart_container(self, client, container_id, timeout=None):
        client_timeout = client.timeout
        if client_timeout is not None:
            client_timeout += timeout or 10
        client.post_call('/containers/{0}/restart', container_id, params={'t': timeout}, timeout=client_timeout)

    def kill_container(self, client, container_id, kill_signal=None):
        params = {}
        if kill_signal is not None:
            params['signal'] = kill_signal
        client.post_call('/containers/{0}/kill', container_id, params=params)

    def stop_container(self, client, container_id, timeout=None):
        if timeout:
            params = {'t': timeout}
        else:
            params = {}
            timeout = 10
        client_timeout = client.timeout
        if client_timeout is not None:
            client_timeout += timeout
        count = 0
        while True:
            try:
                client.post_call('/containers/{0}/stop', container_id, params=params, timeout=client_timeout)
            except APIError as exc:
                if 'Unpause the container before stopping or killing' in exc.explanation:
                    if count == 3:
                        raise Exception('%s [tried to unpause three times]' % to_native(exc))
                    count += 1
                    try:
                        self.unpause_container(client, container_id)
                    except Exception as exc2:
                        raise Exception('%s [while unpausing]' % to_native(exc2))
                    continue
                raise
            break

    def remove_container(self, client, container_id, remove_volumes=False, link=False, force=False):
        params = {'v': remove_volumes, 'link': link, 'force': force}
        count = 0
        while True:
            try:
                client.delete_call('/containers/{0}', container_id, params=params)
            except NotFound as dummy:
                pass
            except APIError as exc:
                if 'Unpause the container before stopping or killing' in exc.explanation:
                    if count == 3:
                        raise Exception('%s [tried to unpause three times]' % to_native(exc))
                    count += 1
                    try:
                        self.unpause_container(client, container_id)
                    except Exception as exc2:
                        raise Exception('%s [while unpausing]' % to_native(exc2))
                    continue
                if 'removal of container ' in exc.explanation and ' is already in progress' in exc.explanation:
                    pass
                else:
                    raise
            break

    def run(self, runner, client):
        try:
            runner()
        except DockerException as e:
            client.fail('An unexpected Docker error occurred: {0}'.format(to_native(e)), exception=traceback.format_exc())
        except RequestException as e:
            client.fail('An unexpected requests error occurred when trying to talk to the Docker daemon: {0}'.format(to_native(e)), exception=traceback.format_exc())