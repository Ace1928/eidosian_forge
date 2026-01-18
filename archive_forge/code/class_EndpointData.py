import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
class EndpointData(object):
    """Normalized information about a discovered endpoint.

    Contains url, version, microversion, interface and region information.
    This is essentially the data contained in the catalog and the version
    discovery documents about an endpoint that is used to select the endpoint
    desired by the user. It is returned so that a user can know which qualities
    a discovered endpoint had, in case their request allowed for a range of
    possibilities.
    """

    def __init__(self, catalog_url=None, service_url=None, service_type=None, service_name=None, service_id=None, region_name=None, interface=None, endpoint_id=None, raw_endpoint=None, api_version=None, major_version=None, min_microversion=None, max_microversion=None, next_min_version=None, not_before=None, status=None):
        self.catalog_url = catalog_url
        self.service_url = service_url
        self.service_type = service_type
        self.service_name = service_name
        self.service_id = service_id
        self.interface = interface
        self.region_name = region_name
        self.endpoint_id = endpoint_id
        self.raw_endpoint = raw_endpoint
        self.major_version = major_version
        self.min_microversion = min_microversion
        self.max_microversion = max_microversion
        self.next_min_version = next_min_version
        self.not_before = not_before
        self.status = status
        self._saved_project_id = None
        self._catalog_matches_version = False
        self._catalog_matches_exactly = False
        self._disc = None
        self.api_version = api_version or _version_from_url(self.url)

    def __copy__(self):
        """Return a new EndpointData based on this one."""
        new_data = EndpointData(catalog_url=self.catalog_url, service_url=self.service_url, service_type=self.service_type, service_name=self.service_name, service_id=self.service_id, region_name=self.region_name, interface=self.interface, endpoint_id=self.endpoint_id, raw_endpoint=self.raw_endpoint, api_version=self.api_version, major_version=self.major_version, min_microversion=self.min_microversion, max_microversion=self.max_microversion, next_min_version=self.next_min_version, not_before=self.not_before, status=self.status)
        new_data._disc = self._disc
        new_data._saved_project_id = self._saved_project_id
        return new_data

    def __str__(self):
        """Produce a string like EndpointData{key=val, ...}, for debugging."""
        str_attrs = ('api_version', 'catalog_url', 'endpoint_id', 'interface', 'major_version', 'max_microversion', 'min_microversion', 'next_min_version', 'not_before', 'raw_endpoint', 'region_name', 'service_id', 'service_name', 'service_type', 'service_url', 'url')
        return '%s{%s}' % (self.__class__.__name__, ', '.join(['%s=%s' % (attr, getattr(self, attr)) for attr in str_attrs]))

    @property
    def url(self):
        return self.service_url or self.catalog_url

    def get_current_versioned_data(self, session, allow=None, cache=None, project_id=None):
        """Run version discovery on the current endpoint.

        A simplified version of get_versioned_data, get_current_versioned_data
        runs discovery but only on the endpoint that has been found already.

        It can be useful in some workflows where the user wants version
        information about the endpoint they have.

        :param session: A session object that can be used for communication.
        :type session: keystoneauth1.session.Session
        :param dict allow: Extra filters to pass when discovering API
                           versions. (optional)
        :param dict cache: A dict to be used for caching results in
                           addition to caching them on the Session.
                           (optional)
        :param string project_id: ID of the currently scoped project. Used for
                                  removing project_id components of URLs from
                                  the catalog. (optional)

        :returns: A new EndpointData with the requested versioned data.
        :rtype: :py:class:`keystoneauth1.discover.EndpointData`
        :raises keystoneauth1.exceptions.discovery.DiscoveryFailure: If the
                                                    appropriate versioned data
                                                    could not be discovered.
        """
        min_version, max_version = _normalize_version_args(self.api_version, None, None)
        return self.get_versioned_data(session=session, allow=allow, cache=cache, allow_version_hack=True, discover_versions=True, min_version=min_version, max_version=max_version)

    def get_versioned_data(self, session, allow=None, cache=None, allow_version_hack=True, project_id=None, discover_versions=True, min_version=None, max_version=None):
        """Run version discovery for the service described.

        Performs Version Discovery and returns a new EndpointData object with
        information found.

        min_version and max_version can be given either as strings or tuples.

        :param session: A session object that can be used for communication.
        :type session: keystoneauth1.session.Session
        :param dict allow: Extra filters to pass when discovering API
                           versions. (optional)
        :param dict cache: A dict to be used for caching results in
                           addition to caching them on the Session.
                           (optional)
        :param bool allow_version_hack: Allow keystoneauth to hack up catalog
                                        URLS to support older schemes.
                                        (optional, default True)
        :param string project_id: ID of the currently scoped project. Used for
                                  removing project_id components of URLs from
                                  the catalog. (optional)
        :param bool discover_versions: Whether to get version metadata from
                                       the version discovery document even
                                       if it's not neccessary to fulfill the
                                       major version request. (optional,
                                       defaults to True)
        :param min_version: The minimum version that is acceptable. If
                            min_version is given with no max_version it is as
                            if max version is 'latest'.
        :param max_version: The maximum version that is acceptable. If
                            min_version is given with no max_version it is as
                            if max version is 'latest'.

        :returns: A new EndpointData with the requested versioned data.
        :rtype: :py:class:`keystoneauth1.discover.EndpointData`
        :raises keystoneauth1.exceptions.discovery.DiscoveryFailure: If the
                                                    appropriate versioned data
                                                    could not be discovered.
        """
        min_version, max_version = _normalize_version_args(None, min_version, max_version)
        if not allow:
            allow = {}
        new_data = copy.copy(self)
        new_data._set_version_info(session=session, allow=allow, cache=cache, allow_version_hack=allow_version_hack, project_id=project_id, discover_versions=discover_versions, min_version=min_version, max_version=max_version)
        return new_data

    def get_all_version_string_data(self, session, project_id=None):
        """Return version data for all versions discovery can find.

        :param string project_id: ID of the currently scoped project. Used for
                                  removing project_id components of URLs from
                                  the catalog. (optional)
        :returns: A list of :class:`VersionData` sorted by version number.
        :rtype: list(VersionData)
        """
        versions = []
        for vers_url in self._get_discovery_url_choices(project_id=project_id):
            try:
                d = get_discovery(session, vers_url)
            except Exception as e:
                _LOGGER.debug('Failed attempt at discovery on %s: %s', vers_url, str(e))
                continue
            for version in d.version_string_data():
                versions.append(version)
            break
        return versions or self._infer_version_data(project_id)

    def _infer_version_data(self, project_id=None):
        """Return version data dict for when discovery fails.

        :param string project_id: ID of the currently scoped project. Used for
                                  removing project_id components of URLs from
                                  the catalog. (optional)
        :returns: A list of :class:`VersionData` sorted by version number.
        :rtype: list(VersionData)
        """
        version = self.api_version
        if version:
            version = version_to_string(self.api_version)
        url = self.url.rstrip('/')
        if project_id and url.endswith(project_id):
            url, _ = self.url.rsplit('/', 1)
        url += '/'
        return [VersionData(url=url, version=version)]

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

    def _get_discovery_url_choices(self, project_id=None, allow_version_hack=True, min_version=None, max_version=None):
        """Find potential locations for version discovery URLs.

        min_version and max_version are already normalized, so will either be
        None or a tuple.
        """
        url = urllib.parse.urlparse(self.url.rstrip('/'))
        url_parts = url.path.split('/')
        if project_id and url_parts[-1] == project_id:
            self._saved_project_id = url_parts.pop()
        elif not project_id:
            try:
                normalize_version_number(url_parts[-2])
                self._saved_project_id = url_parts.pop()
            except (IndexError, TypeError):
                pass
        catalog_discovery = versioned_discovery = None
        try:
            url_version = normalize_version_number(url_parts[-1])
            versioned_discovery = urllib.parse.ParseResult(url.scheme, url.netloc, '/'.join(url_parts), url.params, url.query, url.fragment).geturl()
        except TypeError:
            pass
        else:
            is_between = min_version and max_version and version_between(min_version, max_version, url_version)
            exact_match = is_between and max_version and (max_version[0] == url_version[0])
            high_match = is_between and max_version and (max_version[1] != LATEST) and version_match(max_version, url_version)
            if exact_match or is_between:
                self._catalog_matches_version = True
                self._catalog_matches_exactly = exact_match
                catalog_discovery = urllib.parse.ParseResult(url.scheme, url.netloc, '/'.join(url_parts), url.params, url.query, url.fragment).geturl().rstrip('/') + '/'
            if catalog_discovery and (high_match or exact_match):
                yield catalog_discovery
                catalog_discovery = None
            url_parts.pop()
        if allow_version_hack:
            hacked_url = urllib.parse.ParseResult(url.scheme, url.netloc, '/'.join(url_parts), url.params, url.query, url.fragment).geturl()
            if hacked_url != self.catalog_url:
                hacked_url = hacked_url.strip('/') + '/'
            yield hacked_url
            if catalog_discovery:
                yield catalog_discovery
            yield self._get_catalog_discover_hack()
        elif versioned_discovery and self._saved_project_id:
            yield versioned_discovery
        yield self.catalog_url

    def _get_catalog_discover_hack(self):
        """Apply the catalog hacks and figure out an unversioned endpoint.

        This function is internal to keystoneauth1.

        :returns: A url that has been transformed by the regex hacks that
                  match the service_type.
        """
        return _VERSION_HACKS.get_discover_hack(self.service_type, self.url)