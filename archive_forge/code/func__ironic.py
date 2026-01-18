import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def _ironic(self, action, cmd='ironic', flags='', params='', merge_stderr=False):
    """Execute ironic command for the given action.

        :param action: the cli command to run using Ironic
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
        id_api_version = config.get('os_identity_api_version')
        if id_api_version:
            flags += ' --os-identity-api-version {}'.format(id_api_version)
    else:
        flags += ' --os-endpoint-type publicURL'
    if hasattr(self, 'ironic_url'):
        if cmd == 'openstack':
            flags += ' --os-auth-type none'
        return self._cmd_no_auth(cmd, action, flags, params)
    else:
        for keystone_object in ('user', 'project'):
            domain_attr = 'os_%s_domain_id' % keystone_object
            if hasattr(self, domain_attr):
                flags += ' --os-%(ks_obj)s-domain-id %(value)s' % {'ks_obj': keystone_object, 'value': getattr(self, domain_attr)}
        return self.client.cmd_with_auth(cmd, action, flags, params, merge_stderr=merge_stderr)