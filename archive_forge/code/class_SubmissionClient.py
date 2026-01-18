import dataclasses
import importlib
import logging
import json
import os
import yaml
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union
from pkg_resources import packaging
import ray
import ssl
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray.dashboard.modules.job.common import uri_to_http_components
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.utils import split_address
from ray.autoscaler._private.cli_logger import cli_logger
class SubmissionClient:

    def __init__(self, address: Optional[str]=None, create_cluster_if_needed: bool=False, cookies: Optional[Dict[str, Any]]=None, metadata: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, Any]]=None, verify: Optional[Union[str, bool]]=True):
        if address is not None and address.endswith('/'):
            address = address.rstrip('/')
            logger.debug(f'The submission address cannot contain trailing slashes. Removing them from the requested submission address of "{address}".')
        cluster_info = parse_cluster_info(address, create_cluster_if_needed, cookies, metadata, headers)
        self._address = cluster_info.address
        self._cookies = cluster_info.cookies
        self._default_metadata = cluster_info.metadata or {}
        self._headers = cluster_info.headers
        self._verify = verify
        if isinstance(self._verify, str):
            if os.path.isdir(self._verify):
                cafile, capath = (None, self._verify)
            elif os.path.isfile(self._verify):
                cafile, capath = (self._verify, None)
            else:
                raise FileNotFoundError(f"Path to CA certificates: '{self._verify}', does not exist.")
            self._ssl_context = ssl.create_default_context(cafile=cafile, capath=capath)
        elif self._verify is False:
            self._ssl_context = False
        else:
            self._ssl_context = None

    def _check_connection_and_version(self, min_version: str='1.9', version_error_message: str=None):
        self._check_connection_and_version_with_url(min_version, version_error_message)

    def _check_connection_and_version_with_url(self, min_version: str='1.9', version_error_message: str=None, url: str='/api/version'):
        if version_error_message is None:
            version_error_message = f'Please ensure the cluster is running Ray {min_version} or higher.'
        try:
            r = self._do_request('GET', url)
            if r.status_code == 404:
                raise RuntimeError('Version check returned 404. ' + version_error_message)
            r.raise_for_status()
            running_ray_version = r.json()['ray_version']
            if packaging.version.parse(running_ray_version) < packaging.version.parse(min_version):
                raise RuntimeError(f'Ray version {running_ray_version} is running on the cluster. ' + version_error_message)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f'Failed to connect to Ray at address: {self._address}.')

    def _raise_error(self, r: 'requests.Response'):
        raise RuntimeError(f'Request failed with status code {r.status_code}: {r.text}.')

    def _do_request(self, method: str, endpoint: str, *, data: Optional[bytes]=None, json_data: Optional[dict]=None, **kwargs) -> 'requests.Response':
        """Perform the actual HTTP request

        Keyword arguments other than "cookies", "headers" are forwarded to the
        `requests.request()`.
        """
        url = self._address + endpoint
        logger.debug(f'Sending request to {url} with json data: {json_data or {}}.')
        return requests.request(method, url, cookies=self._cookies, data=data, json=json_data, headers=self._headers, verify=self._verify, **kwargs)

    def _package_exists(self, package_uri: str) -> bool:
        protocol, package_name = uri_to_http_components(package_uri)
        r = self._do_request('GET', f'/api/packages/{protocol}/{package_name}')
        if r.status_code == 200:
            logger.debug(f'Package {package_uri} already exists.')
            return True
        elif r.status_code == 404:
            logger.debug(f'Package {package_uri} does not exist.')
            return False
        else:
            self._raise_error(r)

    def _upload_package(self, package_uri: str, package_path: str, include_parent_dir: Optional[bool]=False, excludes: Optional[List[str]]=None, is_file: bool=False) -> bool:
        logger.info(f'Uploading package {package_uri}.')
        with tempfile.TemporaryDirectory() as tmp_dir:
            protocol, package_name = uri_to_http_components(package_uri)
            if is_file:
                package_file = Path(package_path)
            else:
                package_file = Path(tmp_dir) / package_name
                create_package(package_path, package_file, include_parent_dir=include_parent_dir, excludes=excludes)
            try:
                r = self._do_request('PUT', f'/api/packages/{protocol}/{package_name}', data=package_file.read_bytes())
                if r.status_code != 200:
                    self._raise_error(r)
            finally:
                if not is_file:
                    package_file.unlink()

    def _upload_package_if_needed(self, package_path: str, include_parent_dir: bool=False, excludes: Optional[List[str]]=None, is_file: bool=False) -> str:
        if is_file:
            package_uri = get_uri_for_package(Path(package_path))
        else:
            package_uri = get_uri_for_directory(package_path, excludes=excludes)
        if not self._package_exists(package_uri):
            self._upload_package(package_uri, package_path, include_parent_dir=include_parent_dir, excludes=excludes, is_file=is_file)
        else:
            logger.info(f'Package {package_uri} already exists, skipping upload.')
        return package_uri

    def _upload_working_dir_if_needed(self, runtime_env: Dict[str, Any]):

        def _upload_fn(working_dir, excludes, is_file=False):
            self._upload_package_if_needed(working_dir, include_parent_dir=False, excludes=excludes, is_file=is_file)
        upload_working_dir_if_needed(runtime_env, upload_fn=_upload_fn)

    def _upload_py_modules_if_needed(self, runtime_env: Dict[str, Any]):

        def _upload_fn(module_path, excludes, is_file=False):
            self._upload_package_if_needed(module_path, include_parent_dir=True, excludes=excludes, is_file=is_file)
        upload_py_modules_if_needed(runtime_env, upload_fn=_upload_fn)

    @PublicAPI(stability='beta')
    def get_version(self) -> str:
        r = self._do_request('GET', '/api/version')
        if r.status_code == 200:
            return r.json().get('version')
        else:
            self._raise_error(r)

    @DeveloperAPI
    def get_address(self) -> str:
        return self._address