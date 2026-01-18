import logging
from openstack.config import exceptions as sdk_exceptions
from openstack.config import loader as config
from oslo_utils import strutils
def _auth_v2_ignore_v3(self, config):
    """Remove v3 arguments if present for v2 plugin

        Migrated from clientmanager.setup_auth()
        """
    if str(config.get('identity_api_version', '')).startswith('2') and config.get('auth_type').endswith('password'):
        domain_props = ['project_domain_id', 'project_domain_name', 'user_domain_id', 'user_domain_name']
        for prop in domain_props:
            if config['auth'].pop(prop, None) is not None:
                if config.get('cloud'):
                    LOG.warning('Ignoring domain related config %s for %sbecause identity API version is 2.0' % (prop, config['cloud']))
                else:
                    LOG.warning('Ignoring domain related config %s because identity API version is 2.0' % prop)
    return config