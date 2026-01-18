from __future__ import absolute_import, division, print_function
import collections
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
class NCConfiguration(ConfigBase):

    def __init__(self, module):
        super(NCConfiguration, self).__init__(module)
        self._system_meta = collections.OrderedDict()
        self._system_domain_meta = collections.OrderedDict()
        self._system_server_meta = collections.OrderedDict()
        self._hostname_meta = collections.OrderedDict()
        self._lookup_source_meta = collections.OrderedDict()
        self._lookup_meta = collections.OrderedDict()

    def map_obj_to_xml_rpc(self):
        self._system_meta.update([('vrfs', {'xpath': 'ip-domain/vrfs', 'tag': True, 'operation': 'edit'}), ('vrf', {'xpath': 'ip-domain/vrfs/vrf', 'tag': True, 'operation': 'edit'}), ('a:vrf', {'xpath': 'ip-domain/vrfs/vrf/vrf-name', 'operation': 'edit'}), ('a:domain_name', {'xpath': 'ip-domain/vrfs/vrf/name', 'operation': 'edit', 'attrib': 'operation'})])
        self._system_domain_meta.update([('vrfs', {'xpath': 'ip-domain/vrfs', 'tag': True, 'operation': 'edit'}), ('vrf', {'xpath': 'ip-domain/vrfs/vrf', 'tag': True, 'operation': 'edit'}), ('a:vrf', {'xpath': 'ip-domain/vrfs/vrf/vrf-name', 'operation': 'edit'}), ('lists', {'xpath': 'ip-domain/vrfs/vrf/lists', 'tag': True, 'operation': 'edit'}), ('list', {'xpath': 'ip-domain/vrfs/vrf/lists/list', 'tag': True, 'operation': 'edit', 'attrib': 'operation'}), ('a:order', {'xpath': 'ip-domain/vrfs/vrf/lists/list/order', 'operation': 'edit'}), ('a:domain_search', {'xpath': 'ip-domain/vrfs/vrf/lists/list/list-name', 'operation': 'edit'})])
        self._system_server_meta.update([('vrfs', {'xpath': 'ip-domain/vrfs', 'tag': True, 'operation': 'edit'}), ('vrf', {'xpath': 'ip-domain/vrfs/vrf', 'tag': True, 'operation': 'edit'}), ('a:vrf', {'xpath': 'ip-domain/vrfs/vrf/vrf-name', 'operation': 'edit'}), ('servers', {'xpath': 'ip-domain/vrfs/vrf/servers', 'tag': True, 'operation': 'edit'}), ('server', {'xpath': 'ip-domain/vrfs/vrf/servers/server', 'tag': True, 'operation': 'edit', 'attrib': 'operation'}), ('a:order', {'xpath': 'ip-domain/vrfs/vrf/servers/server/order', 'operation': 'edit'}), ('a:name_servers', {'xpath': 'ip-domain/vrfs/vrf/servers/server/server-address', 'operation': 'edit'})])
        self._hostname_meta.update([('a:hostname', {'xpath': 'host-names/host-name', 'operation': 'edit', 'attrib': 'operation'})])
        self._lookup_source_meta.update([('vrfs', {'xpath': 'ip-domain/vrfs', 'tag': True, 'operation': 'edit'}), ('vrf', {'xpath': 'ip-domain/vrfs/vrf', 'tag': True, 'operation': 'edit'}), ('a:vrf', {'xpath': 'ip-domain/vrfs/vrf/vrf-name', 'operation': 'edit'}), ('a:lookup_source', {'xpath': 'ip-domain/vrfs/vrf/source-interface', 'operation': 'edit', 'attrib': 'operation'})])
        self._lookup_meta.update([('vrfs', {'xpath': 'ip-domain/vrfs', 'tag': True, 'operation': 'edit'}), ('vrf', {'xpath': 'ip-domain/vrfs/vrf', 'tag': True, 'operation': 'edit'}), ('a:vrf', {'xpath': 'ip-domain/vrfs/vrf/vrf-name', 'operation': 'edit'}), ('lookup', {'xpath': 'ip-domain/vrfs/vrf/lookup', 'tag': True, 'operation': 'edit', 'attrib': 'operation'})])
        state = self._module.params['state']
        _get_filter = build_xml('ip-domain', opcode='filter')
        running = get_config(self._module, source='running', config_filter=_get_filter)
        _get_filter = build_xml('host-names', opcode='filter')
        hostname_runn = get_config(self._module, source='running', config_filter=_get_filter)
        hostname_ele = etree_find(hostname_runn, 'host-name')
        hostname = hostname_ele.text if hostname_ele is not None else None
        vrf_ele = etree_findall(running, 'vrf')
        vrf_map = {}
        for vrf in vrf_ele:
            name_server_list = list()
            domain_list = list()
            vrf_name_ele = etree_find(vrf, 'vrf-name')
            vrf_name = vrf_name_ele.text if vrf_name_ele is not None else None
            domain_name_ele = etree_find(vrf, 'name')
            domain_name = domain_name_ele.text if domain_name_ele is not None else None
            domain_ele = etree_findall(vrf, 'list-name')
            for domain in domain_ele:
                domain_list.append(domain.text)
            server_ele = etree_findall(vrf, 'server-address')
            for server in server_ele:
                name_server_list.append(server.text)
            lookup_source_ele = etree_find(vrf, 'source-interface')
            lookup_source = lookup_source_ele.text if lookup_source_ele is not None else None
            lookup_enabled = False if etree_find(vrf, 'lookup') is not None else True
            vrf_map[vrf_name] = {'domain_name': domain_name, 'domain_search': domain_list, 'name_servers': name_server_list, 'lookup_source': lookup_source, 'lookup_enabled': lookup_enabled}
        opcode = None
        hostname_param = {}
        lookup_param = {}
        system_param = {}
        sys_server_params = list()
        sys_domain_params = list()
        add_domain_params = list()
        del_domain_params = list()
        add_server_params = list()
        del_server_params = list()
        lookup_source_params = {}
        try:
            sys_node = vrf_map[self._want['vrf']]
        except KeyError:
            sys_node = {'domain_name': None, 'domain_search': [], 'name_servers': [], 'lookup_source': None, 'lookup_enabled': True}
        if state == 'absent':
            opcode = 'delete'

            def needs_update(x):
                return self._want[x] is not None and self._want[x] == sys_node[x]
            if needs_update('domain_name'):
                system_param = {'vrf': self._want['vrf'], 'domain_name': self._want['domain_name']}
            if needs_update('hostname'):
                hostname_param = {'hostname': hostname}
            if not self._want['lookup_enabled'] and (not sys_node['lookup_enabled']):
                lookup_param['vrf'] = self._want['vrf']
            if needs_update('lookup_source'):
                lookup_source_params['vrf'] = self._want['vrf']
                lookup_source_params['lookup_source'] = self._want['lookup_source']
            if self._want['domain_search']:
                domain_param = {}
                domain_param['domain_name'] = self._want['domain_name']
                domain_param['vrf'] = self._want['vrf']
                domain_param['order'] = '0'
                for domain in self._want['domain_search']:
                    if domain in sys_node['domain_search']:
                        domain_param['domain_search'] = domain
                        sys_domain_params.append(domain_param.copy())
            if self._want['name_servers']:
                server_param = {}
                server_param['vrf'] = self._want['vrf']
                server_param['order'] = '0'
                for server in self._want['name_servers']:
                    if server in sys_node['name_servers']:
                        server_param['name_servers'] = server
                        sys_server_params.append(server_param.copy())
        elif state == 'present':
            opcode = 'merge'

            def needs_update(x):
                return self._want[x] is not None and (sys_node[x] is None or (sys_node[x] is not None and self._want[x] != sys_node[x]))
            if needs_update('domain_name'):
                system_param = {'vrf': self._want['vrf'], 'domain_name': self._want['domain_name']}
            if self._want['hostname'] is not None and self._want['hostname'] != hostname:
                hostname_param = {'hostname': self._want['hostname']}
            if not self._want['lookup_enabled'] and sys_node['lookup_enabled']:
                lookup_param['vrf'] = self._want['vrf']
            if needs_update('lookup_source'):
                lookup_source_params['vrf'] = self._want['vrf']
                lookup_source_params['lookup_source'] = self._want['lookup_source']
            if self._want['domain_search']:
                domain_adds, domain_removes = diff_list(self._want['domain_search'], sys_node['domain_search'])
                domain_param = {}
                domain_param['domain_name'] = self._want['domain_name']
                domain_param['vrf'] = self._want['vrf']
                domain_param['order'] = '0'
                for domain in domain_adds:
                    if domain not in sys_node['domain_search']:
                        domain_param['domain_search'] = domain
                        add_domain_params.append(domain_param.copy())
                for domain in domain_removes:
                    if domain in sys_node['domain_search']:
                        domain_param['domain_search'] = domain
                        del_domain_params.append(domain_param.copy())
            if self._want['name_servers']:
                server_adds, server_removes = diff_list(self._want['name_servers'], sys_node['name_servers'])
                server_param = {}
                server_param['vrf'] = self._want['vrf']
                server_param['order'] = '0'
                for domain in server_adds:
                    if domain not in sys_node['name_servers']:
                        server_param['name_servers'] = domain
                        add_server_params.append(server_param.copy())
                for domain in server_removes:
                    if domain in sys_node['name_servers']:
                        server_param['name_servers'] = domain
                        del_server_params.append(server_param.copy())
        self._result['xml'] = []
        _edit_filter_list = list()
        if opcode:
            if hostname_param:
                _edit_filter_list.append(build_xml('host-names', xmap=self._hostname_meta, params=hostname_param, opcode=opcode))
            if system_param:
                _edit_filter_list.append(build_xml('ip-domain', xmap=self._system_meta, params=system_param, opcode=opcode))
            if lookup_source_params:
                _edit_filter_list.append(build_xml('ip-domain', xmap=self._lookup_source_meta, params=lookup_source_params, opcode=opcode))
            if lookup_param:
                _edit_filter_list.append(build_xml('ip-domain', xmap=self._lookup_meta, params=lookup_param, opcode=opcode))
            if opcode == 'delete':
                if sys_domain_params:
                    _edit_filter_list.append(build_xml('ip-domain', xmap=self._system_domain_meta, params=sys_domain_params, opcode=opcode))
                if sys_server_params:
                    _edit_filter_list.append(build_xml('ip-domain', xmap=self._system_server_meta, params=sys_server_params, opcode=opcode))
                    if self._want['vrf'] != 'default':
                        self._result['warnings'] = ["name-servers delete operation with non-default vrf is a success, but with rpc-error. Recommended to use 'ignore_errors' option with the task as a workaround"]
            elif opcode == 'merge':
                if add_domain_params:
                    _edit_filter_list.append(build_xml('ip-domain', xmap=self._system_domain_meta, params=add_domain_params, opcode=opcode))
                if del_domain_params:
                    _edit_filter_list.append(build_xml('ip-domain', xmap=self._system_domain_meta, params=del_domain_params, opcode='delete'))
                if add_server_params:
                    _edit_filter_list.append(build_xml('ip-domain', xmap=self._system_server_meta, params=add_server_params, opcode=opcode))
                if del_server_params:
                    _edit_filter_list.append(build_xml('ip-domain', xmap=self._system_server_meta, params=del_server_params, opcode='delete'))
        diff = None
        if _edit_filter_list:
            commit = not self._module.check_mode
            diff = load_config(self._module, _edit_filter_list, commit=commit, running=running, nc_get_filter=_get_filter)
        if diff:
            if self._module._diff:
                self._result['diff'] = dict(prepared=diff)
            self._result['xml'] = _edit_filter_list
            self._result['changed'] = True

    def run(self):
        self.map_params_to_obj()
        self.map_obj_to_xml_rpc()
        return self._result