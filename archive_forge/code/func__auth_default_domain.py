import logging
from openstack.config import exceptions as sdk_exceptions
from openstack.config import loader as config
from oslo_utils import strutils
def _auth_default_domain(self, config):
    """Set a default domain from available arguments

        Migrated from clientmanager.setup_auth()
        """
    identity_version = str(config.get('identity_api_version', ''))
    auth_type = config.get('auth_type', None)
    default_domain = config.get('default_domain', None)
    if identity_version == '3' and (not auth_type.startswith('v2')) and default_domain:
        if auth_type in ('password', 'v3password', 'v3totp') and (not config['auth'].get('project_domain_id')) and (not config['auth'].get('project_domain_name')):
            config['auth']['project_domain_id'] = default_domain
        if auth_type in ('password', 'v3password', 'v3totp') and (not config['auth'].get('user_domain_id')) and (not config['auth'].get('user_domain_name')):
            config['auth']['user_domain_id'] = default_domain
    return config