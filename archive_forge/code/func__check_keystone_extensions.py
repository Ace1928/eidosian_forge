import logging
import urllib.parse as urlparse
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient.i18n import _
def _check_keystone_extensions(self, url):
    """Call Keystone URL and detects the available extensions."""
    try:
        if not url.endswith('/'):
            url += '/'
        resp, body = self._request('%sextensions' % url, 'GET', headers={'Accept': 'application/json'})
        if resp.status_code in (200, 204):
            if 'extensions' in body and 'values' in body['extensions']:
                extensions = body['extensions']['values']
            elif 'extensions' in body:
                extensions = body['extensions']
            else:
                return dict(message=_('Unrecognized extensions response from %s') % url)
            return dict((self._get_extension_info(e) for e in extensions))
        elif resp.status_code == 305:
            return self._check_keystone_extensions(resp['location'])
        else:
            raise exceptions.from_response(resp, 'GET', '%sextensions' % url)
    except Exception:
        _logger.exception('Failed to check keystone extensions.')