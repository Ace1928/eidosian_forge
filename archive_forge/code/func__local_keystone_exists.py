import logging
import urllib.parse as urlparse
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient.i18n import _
def _local_keystone_exists(self):
    """Check if Keystone is available on default local port 35357."""
    results = self._check_keystone_versions('http://localhost:35357')
    if results is None:
        results = self._check_keystone_versions('https://localhost:35357')
    return results