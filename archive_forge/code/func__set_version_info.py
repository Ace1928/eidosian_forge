import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def _set_version_info(self, session, allow=None, cache=None, allow_version_hack=True, project_id=None, discover_versions=False, min_version=None, max_version=None):
    match_url = None
    no_version = not max_version and (not min_version)
    if no_version and (not discover_versions):
        return
    elif no_version and discover_versions:
        allow_version_hack = False
        match_url = self.url
    if project_id:
        self.project_id = project_id
    discovered_data = None
    if self._disc:
        discovered_data = self._disc.versioned_data_for(min_version=min_version, max_version=max_version, url=match_url, **allow)
    if not discovered_data:
        self._run_discovery(session=session, cache=cache, min_version=min_version, max_version=max_version, project_id=project_id, allow_version_hack=allow_version_hack, discover_versions=discover_versions)
        if not self._disc:
            return
        discovered_data = self._disc.versioned_data_for(min_version=min_version, max_version=max_version, url=match_url, **allow)
    if not discovered_data:
        if min_version and (not max_version):
            raise exceptions.DiscoveryFailure('Minimum version {min_version} was not found'.format(min_version=version_to_string(min_version)))
        elif max_version and (not min_version):
            raise exceptions.DiscoveryFailure('Maximum version {max_version} was not found'.format(max_version=version_to_string(max_version)))
        elif min_version and max_version:
            raise exceptions.DiscoveryFailure('No version found between {min_version} and {max_version}'.format(min_version=version_to_string(min_version), max_version=version_to_string(max_version)))
        else:
            raise exceptions.DiscoveryFailure('No version data found remotely at all')
    self.min_microversion = discovered_data['min_microversion']
    self.max_microversion = discovered_data['max_microversion']
    self.next_min_version = discovered_data['next_min_version']
    self.not_before = discovered_data['not_before']
    self.api_version = discovered_data['version']
    self.status = discovered_data['status']
    discovered_url = discovered_data['url']
    url = urllib.parse.urljoin(self._disc._url.rstrip('/') + '/', discovered_url)
    if self._saved_project_id:
        url = urllib.parse.urljoin(url.rstrip('/') + '/', self._saved_project_id)
    self.service_url = url