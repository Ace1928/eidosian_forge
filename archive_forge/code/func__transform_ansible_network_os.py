from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.ansible.utils.plugins.plugin_utils.base.cli_parser import CliParserBase
def _transform_ansible_network_os(self):
    """Transform the ansible_network_os to a pyats OS
        The last part of the fully qualified name is used
        org.name.platform => platform

        In the case of ios, the os is assumed to be iosxe
        """
    ane = self._task_vars.get('ansible_network_os', '').split('.')[-1]
    if ane == 'ios':
        self._debug('ansible_network_os was ios, using iosxe.')
        ane = 'iosxe'
    self._debug("OS set to '{ane}' using 'ansible_network_os'.".format(ane=ane))
    return ane