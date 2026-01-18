from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.bgp_global import (
def _compare_rpki_server(self, want, have):
    """Leverages the base class `compare()` method and
        populates the list of commands to be run by comparing
        the `want` and `have` data with the `parsers` defined
        for the Bgp_global rpki servers resource.
        """
    rpki_server_parsers = ['rpki_server_purge_time', 'rpki_server_refresh_time', 'rpki_server_refresh_time_off', 'rpki_server_response_time', 'rpki_server_response_time_off', 'rpki_server_shutdown', 'rpki_server_transport_ssh', 'rpki_server_transport_tcp']
    want = want.get('rpki', {}).get('servers', {})
    have = have.get('rpki', {}).get('servers', {})
    for name, entry in iteritems(want):
        new_have = have.pop(name, {})
        begin = len(self.commands)
        self.compare(parsers=rpki_server_parsers, want=entry, have=new_have)
        rpki_server_name = entry.get('name')
        if len(self.commands) != begin:
            self.commands.insert(begin, self._tmplt.render({'name': rpki_server_name}, 'rpki_server_name', False))
    for name, entry in iteritems(have):
        self.addcmd(entry, 'rpki_server_name', True)