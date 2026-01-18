from __future__ import absolute_import, division, print_function
import abc
import os
import re
import shlex
from functools import partial
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
class EngineDriver(object):
    name = None

    @abc.abstractmethod
    def setup(self, argument_spec, mutually_exclusive=None, required_together=None, required_one_of=None, required_if=None, required_by=None):
        pass

    @abc.abstractmethod
    def get_host_info(self, client):
        pass

    @abc.abstractmethod
    def get_api_version(self, client):
        pass

    @abc.abstractmethod
    def get_container_id(self, container):
        pass

    @abc.abstractmethod
    def get_image_from_container(self, container):
        pass

    @abc.abstractmethod
    def get_image_name_from_container(self, container):
        pass

    @abc.abstractmethod
    def is_container_removing(self, container):
        pass

    @abc.abstractmethod
    def is_container_running(self, container):
        pass

    @abc.abstractmethod
    def is_container_paused(self, container):
        pass

    @abc.abstractmethod
    def inspect_container_by_name(self, client, container_name):
        pass

    @abc.abstractmethod
    def inspect_container_by_id(self, client, container_id):
        pass

    @abc.abstractmethod
    def inspect_image_by_id(self, client, image_id):
        pass

    @abc.abstractmethod
    def inspect_image_by_name(self, client, repository, tag):
        pass

    @abc.abstractmethod
    def pull_image(self, client, repository, tag, platform=None):
        pass

    @abc.abstractmethod
    def pause_container(self, client, container_id):
        pass

    @abc.abstractmethod
    def unpause_container(self, client, container_id):
        pass

    @abc.abstractmethod
    def disconnect_container_from_network(self, client, container_id, network_id):
        pass

    @abc.abstractmethod
    def connect_container_to_network(self, client, container_id, network_id, parameters=None):
        pass

    @abc.abstractmethod
    def create_container(self, client, container_name, create_parameters):
        pass

    @abc.abstractmethod
    def start_container(self, client, container_id):
        pass

    @abc.abstractmethod
    def wait_for_container(self, client, container_id, timeout=None):
        pass

    @abc.abstractmethod
    def get_container_output(self, client, container_id):
        pass

    @abc.abstractmethod
    def update_container(self, client, container_id, update_parameters):
        pass

    @abc.abstractmethod
    def restart_container(self, client, container_id, timeout=None):
        pass

    @abc.abstractmethod
    def kill_container(self, client, container_id, kill_signal=None):
        pass

    @abc.abstractmethod
    def stop_container(self, client, container_id, timeout=None):
        pass

    @abc.abstractmethod
    def remove_container(self, client, container_id, remove_volumes=False, link=False, force=False):
        pass

    @abc.abstractmethod
    def run(self, runner, client):
        pass