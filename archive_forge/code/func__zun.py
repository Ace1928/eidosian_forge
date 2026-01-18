import configparser as config_parser
import os
from tempest.lib.cli import base
def _zun(self, action, cmd='zun', flags='', params='', merge_stderr=False):
    """Execute Zun command for the given action.

        :param action: the cli command to run using Zun
        :type action: string
        :param cmd: the base of cli command to run
        :type action: string
        :param flags: any optional cli flags to use
        :type flags: string
        :param params: any optional positional args to use
        :type params: string
        :param merge_stderr: whether to merge stderr into the result
        :type merge_stderr: bool
        """
    if cmd == 'openstack':
        config = self._get_config()
        id_api_version = config['os_identity_api_version']
        flags += ' --os-identity-api-version {0}'.format(id_api_version)
    else:
        flags += ' --os-endpoint-type publicURL'
    if hasattr(self, 'os_auth_token'):
        return self._cmd_no_auth(cmd, action, flags, params)
    else:
        for keystone_object in ('user', 'project'):
            domain_attr = 'os_%s_domain_id' % keystone_object
            if hasattr(self, domain_attr):
                flags += ' --os-%(ks_obj)s-domain-id %(value)s' % {'ks_obj': keystone_object, 'value': getattr(self, domain_attr)}
        return self.client.cmd_with_auth(cmd, action, flags, params, merge_stderr=merge_stderr)