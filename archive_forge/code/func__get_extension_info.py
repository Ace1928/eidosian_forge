import logging
import urllib.parse as urlparse
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient.i18n import _
@staticmethod
def _get_extension_info(extension):
    """Parse extension information.

        :param extension: a dict of a Keystone extension response
        :returns: tuple - (alias, name)
        """
    alias = extension['alias']
    name = extension['name']
    return (alias, name)