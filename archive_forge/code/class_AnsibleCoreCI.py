from __future__ import annotations
import abc
import dataclasses
import json
import os
import re
import stat
import traceback
import uuid
import time
import typing as t
from .http import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .ci import (
from .data import (
class AnsibleCoreCI:
    """Client for Ansible Core CI services."""
    DEFAULT_ENDPOINT = 'https://ansible-core-ci.testing.ansible.com'

    def __init__(self, args: EnvironmentConfig, resource: Resource, load: bool=True) -> None:
        self.args = args
        self.resource = resource
        self.platform, self.version, self.arch, self.provider = self.resource.as_tuple()
        self.stage = args.remote_stage
        self.client = HttpClient(args)
        self.connection = None
        self.instance_id = None
        self.endpoint = None
        self.default_endpoint = args.remote_endpoint or self.DEFAULT_ENDPOINT
        self.retries = 3
        self.ci_provider = get_ci_provider()
        self.label = self.resource.get_label()
        stripped_label = re.sub('[^A-Za-z0-9_.]+', '-', self.label).strip('-')
        self.name = f'{stripped_label}-{self.stage}'
        self.path = os.path.expanduser(f'~/.ansible/test/instances/{self.name}')
        self.ssh_key = SshKey(args)
        if self.resource.persist and load and self._load():
            try:
                display.info(f'Checking existing {self.label} instance using: {self._uri}', verbosity=1)
                self.connection = self.get(always_raise_on=[404])
                display.info(f'Loaded existing {self.label} instance.', verbosity=1)
            except HttpError as ex:
                if ex.status != 404:
                    raise
                self._clear()
                display.info(f'Cleared stale {self.label} instance.', verbosity=1)
                self.instance_id = None
                self.endpoint = None
        elif not self.resource.persist:
            self.instance_id = None
            self.endpoint = None
            self._clear()
        if self.instance_id:
            self.started: bool = True
        else:
            self.started = False
            self.instance_id = str(uuid.uuid4())
            self.endpoint = None
            display.sensitive.add(self.instance_id)
        if not self.endpoint:
            self.endpoint = self.default_endpoint

    @property
    def available(self) -> bool:
        """Return True if Ansible Core CI is supported."""
        return self.ci_provider.supports_core_ci_auth()

    def start(self) -> t.Optional[dict[str, t.Any]]:
        """Start instance."""
        if self.started:
            display.info(f'Skipping started {self.label} instance.', verbosity=1)
            return None
        return self._start(self.ci_provider.prepare_core_ci_auth())

    def stop(self) -> None:
        """Stop instance."""
        if not self.started:
            display.info(f'Skipping invalid {self.label} instance.', verbosity=1)
            return
        response = self.client.delete(self._uri)
        if response.status_code == 404:
            self._clear()
            display.info(f'Cleared invalid {self.label} instance.', verbosity=1)
            return
        if response.status_code == 200:
            self._clear()
            display.info(f'Stopped running {self.label} instance.', verbosity=1)
            return
        raise self._create_http_error(response)

    def get(self, tries: int=3, sleep: int=15, always_raise_on: t.Optional[list[int]]=None) -> t.Optional[InstanceConnection]:
        """Get instance connection information."""
        if not self.started:
            display.info(f'Skipping invalid {self.label} instance.', verbosity=1)
            return None
        if not always_raise_on:
            always_raise_on = []
        if self.connection and self.connection.running:
            return self.connection
        while True:
            tries -= 1
            response = self.client.get(self._uri)
            if response.status_code == 200:
                break
            error = self._create_http_error(response)
            if not tries or response.status_code in always_raise_on:
                raise error
            display.warning(f'{error}. Trying again after {sleep} seconds.')
            time.sleep(sleep)
        if self.args.explain:
            self.connection = InstanceConnection(running=True, hostname='cloud.example.com', port=12345, username='root', password='password' if self.platform == 'windows' else None)
        else:
            response_json = response.json()
            status = response_json['status']
            con = response_json.get('connection')
            if con:
                self.connection = InstanceConnection(running=status == 'running', hostname=con['hostname'], port=int(con['port']), username=con['username'], password=con.get('password'), response_json=response_json)
            else:
                self.connection = InstanceConnection(running=status == 'running', response_json=response_json)
        if self.connection.password:
            display.sensitive.add(str(self.connection.password))
        status = 'running' if self.connection.running else 'starting'
        display.info(f'The {self.label} instance is {status}.', verbosity=1)
        return self.connection

    def wait(self, iterations: t.Optional[int]=90) -> None:
        """Wait for the instance to become ready."""
        for _iteration in range(1, iterations):
            if self.get().running:
                return
            time.sleep(10)
        raise ApplicationError(f'Timeout waiting for {self.label} instance.')

    @property
    def _uri(self) -> str:
        return f'{self.endpoint}/{self.stage}/{self.provider}/{self.instance_id}'

    def _start(self, auth) -> dict[str, t.Any]:
        """Start instance."""
        display.info(f'Initializing new {self.label} instance using: {self._uri}', verbosity=1)
        data = dict(config=dict(platform=self.platform, version=self.version, architecture=self.arch, public_key=self.ssh_key.pub_contents))
        data.update(auth=auth)
        headers = {'Content-Type': 'application/json'}
        response = self._start_endpoint(data, headers)
        self.started = True
        self._save()
        display.info(f'Started {self.label} instance.', verbosity=1)
        if self.args.explain:
            return {}
        return response.json()

    def _start_endpoint(self, data: dict[str, t.Any], headers: dict[str, str]) -> HttpResponse:
        tries = self.retries
        sleep = 15
        while True:
            tries -= 1
            response = self.client.put(self._uri, data=json.dumps(data), headers=headers)
            if response.status_code == 200:
                return response
            error = self._create_http_error(response)
            if response.status_code == 503:
                raise error
            if not tries:
                raise error
            display.warning(f'{error}. Trying again after {sleep} seconds.')
            time.sleep(sleep)

    def _clear(self) -> None:
        """Clear instance information."""
        try:
            self.connection = None
            os.remove(self.path)
        except FileNotFoundError:
            pass

    def _load(self) -> bool:
        """Load instance information."""
        try:
            data = read_text_file(self.path)
        except FileNotFoundError:
            return False
        if not data.startswith('{'):
            return False
        config = json.loads(data)
        return self.load(config)

    def load(self, config: dict[str, str]) -> bool:
        """Load the instance from the provided dictionary."""
        self.instance_id = str(config['instance_id'])
        self.endpoint = config['endpoint']
        self.started = True
        display.sensitive.add(self.instance_id)
        return True

    def _save(self) -> None:
        """Save instance information."""
        if self.args.explain:
            return
        config = self.save()
        write_json_file(self.path, config, create_directories=True)

    def save(self) -> dict[str, str]:
        """Save instance details and return as a dictionary."""
        return dict(label=self.resource.get_label(), instance_id=self.instance_id, endpoint=self.endpoint)

    @staticmethod
    def _create_http_error(response: HttpResponse) -> ApplicationError:
        """Return an exception created from the given HTTP response."""
        response_json = response.json()
        stack_trace = ''
        if 'message' in response_json:
            message = response_json['message']
        elif 'errorMessage' in response_json:
            message = response_json['errorMessage'].strip()
            if 'stackTrace' in response_json:
                traceback_lines = response_json['stackTrace']
                if traceback_lines and isinstance(traceback_lines[0], list):
                    traceback_lines = traceback.format_list(traceback_lines)
                trace = '\n'.join([x.rstrip() for x in traceback_lines])
                stack_trace = f'\nTraceback (from remote server):\n{trace}'
        else:
            message = str(response_json)
        return CoreHttpError(response.status_code, message, stack_trace)