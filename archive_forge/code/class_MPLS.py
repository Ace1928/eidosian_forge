from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import run_commands
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import ironware_argument_spec, check_args
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
class MPLS(FactsBase):
    COMMANDS = ['show mpls lsp detail', 'show mpls vll-local detail', 'show mpls vll detail', 'show mpls vpls detail']

    def populate(self):
        super(MPLS, self).populate()
        data = self.responses[0]
        if data:
            data = self.parse_mpls(data)
            self.facts['mpls_lsps'] = self.populate_lsps(data)
        data = self.responses[1]
        if data:
            data = self.parse_mpls(data)
            self.facts['mpls_vll_local'] = self.populate_vll_local(data)
        data = self.responses[2]
        if data:
            data = self.parse_mpls(data)
            self.facts['mpls_vll'] = self.populate_vll(data)
        data = self.responses[3]
        if data:
            data = self.parse_mpls(data)
            self.facts['mpls_vpls'] = self.populate_vpls(data)

    def parse_mpls(self, data):
        parsed = dict()
        for line in data.split('\n'):
            if not line:
                continue
            elif line[0] == ' ':
                parsed[key] += '\n%s' % line
            else:
                match = re.match('^(LSP|VLL|VPLS) ([^\\s,]+)', line)
                if match:
                    key = match.group(2)
                    parsed[key] = line
        return parsed

    def populate_vpls(self, vpls):
        facts = dict()
        for key, value in iteritems(vpls):
            vpls = dict()
            vpls['endpoints'] = self.parse_vpls_endpoints(value)
            vpls['vc-id'] = self.parse_vpls_vcid(value)
            facts[key] = vpls
        return facts

    def populate_vll_local(self, vll_locals):
        facts = dict()
        for key, value in iteritems(vll_locals):
            vll = dict()
            vll['endpoints'] = self.parse_vll_endpoints(value)
            facts[key] = vll
        return facts

    def populate_vll(self, vlls):
        facts = dict()
        for key, value in iteritems(vlls):
            vll = dict()
            vll['endpoints'] = self.parse_vll_endpoints(value)
            vll['vc-id'] = self.parse_vll_vcid(value)
            vll['cos'] = self.parse_vll_cos(value)
            facts[key] = vll
        return facts

    def parse_vll_vcid(self, data):
        match = re.search('VC-ID (\\d+),', data, re.M)
        if match:
            return match.group(1)

    def parse_vll_cos(self, data):
        match = re.search('COS +: +(\\d+)', data, re.M)
        if match:
            return match.group(1)

    def parse_vll_endpoints(self, data):
        facts = list()
        regex = 'End-point[0-9 ]*: +(?P<tagged>tagged|untagged) +(vlan +(?P<vlan>[0-9]+) +)?(inner- vlan +(?P<innervlan>[0-9]+) +)?(?P<port>e [0-9/]+|--)'
        matches = re.finditer(regex, data, re.IGNORECASE | re.DOTALL)
        for match in matches:
            f = match.groupdict()
            f['type'] = 'local'
            facts.append(f)
        regex = 'Vll-Peer +: +(?P<vllpeer>[0-9\\.]+).*Tunnel LSP +: +(?P<lsp>\\S+)'
        matches = re.finditer(regex, data, re.IGNORECASE | re.DOTALL)
        for match in matches:
            f = match.groupdict()
            f['type'] = 'remote'
            facts.append(f)
        return facts

    def parse_vpls_vcid(self, data):
        match = re.search('Id (\\d+),', data, re.M)
        if match:
            return match.group(1)

    def parse_vpls_endpoints(self, data):
        facts = list()
        regex = 'Vlan (?P<vlanid>[0-9]+)\\s(?: +(?:L2.*)\\s| +Tagged: (?P<tagged>.+)+\\s| +Untagged: (?P<untagged>.+)\\s)*'
        matches = re.finditer(regex, data, re.IGNORECASE)
        for match in matches:
            f = match.groupdict()
            f['type'] = 'local'
            facts.append(f)
        regex = 'Peer address: (?P<vllpeer>[0-9\\.]+)'
        matches = re.finditer(regex, data, re.IGNORECASE)
        for match in matches:
            f = match.groupdict()
            f['type'] = 'remote'
            facts.append(f)
        return facts

    def populate_lsps(self, lsps):
        facts = dict()
        for key, value in iteritems(lsps):
            lsp = dict()
            lsp['to'] = self.parse_lsp_to(value)
            lsp['from'] = self.parse_lsp_from(value)
            lsp['adminstatus'] = self.parse_lsp_adminstatus(value)
            lsp['operstatus'] = self.parse_lsp_operstatus(value)
            lsp['pri_path'] = self.parse_lsp_pripath(value)
            lsp['sec_path'] = self.parse_lsp_secpath(value)
            lsp['frr'] = self.parse_lsp_frr(value)
            facts[key] = lsp
        return facts

    def parse_lsp_to(self, data):
        match = re.search('^LSP .* to (\\S+)', data, re.M)
        if match:
            return match.group(1)

    def parse_lsp_from(self, data):
        match = re.search('From: ([^\\s,]+),', data, re.M)
        if match:
            return match.group(1)

    def parse_lsp_adminstatus(self, data):
        match = re.search('admin: (\\w+),', data, re.M)
        if match:
            return match.group(1)

    def parse_lsp_operstatus(self, data):
        match = re.search('From: .* status: (\\w+)', data, re.M)
        if match:
            return match.group(1)

    def parse_lsp_pripath(self, data):
        match = re.search('Pri\\. path: ([^\\s,]+), up: (\\w+), active: (\\w+)', data, re.M)
        if match:
            path = dict()
            path['name'] = match.group(1) if match.group(1) != 'NONE' else None
            path['up'] = True if match.group(2) == 'yes' else False
            path['active'] = True if match.group(3) == 'yes' else False
            return path

    def parse_lsp_secpath(self, data):
        match = re.search('Sec\\. path: ([^\\s,]+), active: (\\w+).*\\n.* status: (\\w+)', data, re.M)
        if match:
            path = dict()
            path['name'] = match.group(1) if match.group(1) != 'NONE' else None
            path['up'] = True if match.group(3) == 'up' else False
            path['active'] = True if match.group(2) == 'yes' else False
            return path

    def parse_lsp_frr(self, data):
        match = re.search('Backup LSP: (\\w+)', data, re.M)
        if match:
            path = dict()
            path['up'] = True if match.group(1) == 'UP' else False
            path['name'] = None
            if path['up']:
                match = re.search('bypass_lsp: (\\S)', data, re.M)
                path['name'] = match.group(1) if match else None
            return path