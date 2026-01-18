from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.ntp_global import (
class Ntp_global(ResourceModule):
    """
    The vyos_ntp config class
    """

    def __init__(self, module):
        super(Ntp_global, self).__init__(empty_fact_val={}, facts_module=Facts(module), module=module, resource='ntp_global', tmplt=NtpTemplate())
        self.parsers = ['allow_clients', 'listen_addresses', 'server', 'options', 'allow_clients_delete', 'listen_addresses_delete']

    def execute_module(self):
        """Execute the module

        :rtype: A dictionary
        :returns: The result from module execution
        """
        if self.state not in ['parsed', 'gathered']:
            self.generate_commands()
            self.run_commands()
        return self.result

    def generate_commands(self):
        """Generate configuration commands to send based on
        want, have and desired state.
        """
        wantd = self._ntp_list_to_dict(self.want)
        haved = self._ntp_list_to_dict(self.have)
        if self.state == 'merged':
            wantd = dict_merge(haved, wantd)
        if self.state == 'deleted':
            haved = {k: v for k, v in iteritems(haved) if k in wantd or not wantd}
            wantd = {}
            commandlist = self._commandlist(haved)
            servernames = self._servernames(haved)
            for k, have in iteritems(haved):
                if k not in wantd:
                    for hk, hval in iteritems(have):
                        if hk == 'allow_clients' and hk in commandlist:
                            self.commands.append(self._tmplt.render({'': hk}, 'allow_clients_delete', True))
                            commandlist.remove(hk)
                        elif hk == 'listen_addresses' and hk in commandlist:
                            self.commands.append(self._tmplt.render({'': hk}, 'listen_addresses_delete', True))
                            commandlist.remove(hk)
                        elif hk == 'server' and have['server'] in servernames:
                            self._compareoverride(want={}, have=have)
                            servernames.remove(have['server'])
        if self.state in ['overridden', 'replaced']:
            commandlist = self._commandlist(haved)
            servernames = self._servernames(haved)
            for k, have in iteritems(haved):
                if k not in wantd and 'server' not in have:
                    self._compareoverride(want={}, have=have)
                elif k not in wantd and have['server'] in servernames:
                    self._compareoverride(want={}, have=have)
                    servernames.remove(have['server'])
        for k, want in iteritems(wantd):
            self._compare(want=want, have=haved.pop(k, {}))

    def _compare(self, want, have):
        """Leverages the base class `compare()` method and
        populates the list of commands to be run by comparing
        the `want` and `have` data with the `parsers` defined
        for the Ntp network resource.
        """
        if 'options' in want:
            self.compare(parsers='options', want=want, have=have)
        else:
            self.compare(parsers=self.parsers, want=want, have=have)

    def _compareoverride(self, want, have):
        for i, val in iteritems(have):
            if i == 'options':
                pass
            else:
                self.compare(parsers=i, want={}, have=have)

    def _ntp_list_to_dict(self, entry):
        servers_dict = {}
        for k, data in iteritems(entry):
            if k == 'servers':
                for value in data:
                    if 'options' in value:
                        result = self._serveroptions_list_to_dict(value)
                        for res, resvalue in iteritems(result):
                            servers_dict.update({res: resvalue})
                    else:
                        servers_dict.update({value['server']: value})
            else:
                for value in data:
                    servers_dict.update({'ip_' + value: {k: value}})
        return servers_dict

    def _serveroptions_list_to_dict(self, entry):
        serveroptions_dict = {}
        for Opk, Op in iteritems(entry):
            if Opk == 'options':
                for val in Op:
                    dict = {}
                    dict.update({'server': entry['server']})
                    dict.update({Opk: val})
                    serveroptions_dict.update({entry['server'] + '_' + val: dict})
        return serveroptions_dict

    def _commandlist(self, haved):
        commandlist = []
        for k, have in iteritems(haved):
            for ck, cval in iteritems(have):
                if ck != 'options' and ck not in commandlist:
                    commandlist.append(ck)
        return commandlist

    def _servernames(self, haved):
        servernames = []
        for k, have in iteritems(haved):
            for sk, sval in iteritems(have):
                if sk == 'server' and sval not in ['0.pool.ntp.org', '1.pool.ntp.org', '2.pool.ntp.org']:
                    if sval not in servernames:
                        servernames.append(sval)
        return servernames