import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def _run_discovery(self, session, cache, min_version, max_version, project_id, allow_version_hack, discover_versions):
    tried = set()
    for vers_url in self._get_discovery_url_choices(project_id=project_id, allow_version_hack=allow_version_hack, min_version=min_version, max_version=max_version):
        if self._catalog_matches_exactly and (not discover_versions):
            return
        if vers_url in tried:
            continue
        tried.add(vers_url)
        try:
            self._disc = get_discovery(session, vers_url, cache=cache, authenticated=False)
            break
        except (exceptions.DiscoveryFailure, exceptions.HttpError, exceptions.ConnectionError) as exc:
            _LOGGER.debug('No version document at %s: %s', vers_url, exc)
            continue
    if not self._disc:
        if self._catalog_matches_version:
            self.service_url = self.catalog_url
            return
        if allow_version_hack:
            _LOGGER.warning('Failed to contact the endpoint at %s for discovery. Fallback to using that endpoint as the base url.', self.url)
            return
        else:
            raise exceptions.DiscoveryFailure('Unable to find a version discovery document at %s, the service is unavailable or misconfigured. Required version range (%s - %s), version hack disabled.' % (self.url, min_version or 'any', max_version or 'any'))