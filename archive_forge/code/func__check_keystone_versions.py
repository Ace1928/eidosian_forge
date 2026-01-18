import logging
import urllib.parse as urlparse
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient.i18n import _
def _check_keystone_versions(self, url):
    """Call Keystone URL and detects the available API versions."""
    try:
        resp, body = self._request(url, 'GET', headers={'Accept': 'application/json'})
        if resp.status_code in (200, 204, 300):
            try:
                results = {}
                if 'version' in body:
                    results['message'] = _('Keystone found at %s') % url
                    version = body['version']
                    id, status, version_url = self._get_version_info(version, url)
                    results[str(id)] = {'id': id, 'status': status, 'url': version_url}
                    return results
                elif 'versions' in body:
                    results['message'] = _('Keystone found at %s') % url
                    for version in body['versions']['values']:
                        id, status, version_url = self._get_version_info(version, url)
                        results[str(id)] = {'id': id, 'status': status, 'url': version_url}
                    return results
                else:
                    results['message'] = _('Unrecognized response from %s') % url
                return results
            except KeyError:
                raise exceptions.AuthorizationFailure()
        elif resp.status_code == 305:
            return self._check_keystone_versions(resp['location'])
        else:
            raise exceptions.from_response(resp, 'GET', url)
    except Exception:
        _logger.exception('Failed to detect available versions.')