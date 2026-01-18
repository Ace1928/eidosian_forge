import logging
from openstack.config import exceptions as sdk_exceptions
from openstack.config import loader as config
from oslo_utils import strutils
def _auth_select_default_plugin(self, config):
    """Select a default plugin based on supplied arguments

        Migrated from auth.select_auth_plugin()
        """
    identity_version = str(config.get('identity_api_version', ''))
    if config.get('username', None) and (not config.get('auth_type', None)):
        if identity_version == '3':
            config['auth_type'] = 'v3password'
        elif identity_version.startswith('2'):
            config['auth_type'] = 'v2password'
        else:
            config['auth_type'] = 'password'
    elif config.get('token', None) and (not config.get('auth_type', None)):
        if identity_version == '3':
            config['auth_type'] = 'v3token'
        elif identity_version.startswith('2'):
            config['auth_type'] = 'v2token'
        else:
            config['auth_type'] = 'token'
    elif not config.get('auth_type', None):
        config['auth_type'] = 'password'
    LOG.debug('Auth plugin %s selected' % config['auth_type'])
    return config