from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
class EvpnBgp(object):
    """
    Manages evpn bgp configuration.
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.netconf = None
        self.init_module()
        self.as_number = self.module.params['as_number']
        self.bgp_instance = self.module.params['bgp_instance']
        self.peer_address = self.module.params['peer_address']
        self.peer_group_name = self.module.params['peer_group_name']
        self.peer_enable = self.module.params['peer_enable']
        self.advertise_router_type = self.module.params['advertise_router_type']
        self.vpn_name = self.module.params['vpn_name']
        self.advertise_l2vpn_evpn = self.module.params['advertise_l2vpn_evpn']
        self.state = self.module.params['state']
        self.host = self.module.params['host']
        self.username = self.module.params['username']
        self.port = self.module.params['port']
        self.config = ''
        self.config_list = list()
        self.l2vpn_evpn_exist = False
        self.changed = False
        self.updates_cmd = list()
        self.commands = list()
        self.results = dict()
        self.existing = dict()
        self.proposed = dict()
        self.end_state = dict()

    def init_module(self):
        """ init module """
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def get_evpn_overlay_config(self):
        """get evpn-overlay enable configuration"""
        cmd = 'display current-configuration | include ^evpn-overlay enable'
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        return out

    def get_current_config(self):
        """get current configuration"""
        cmd = 'display current-configuration | section include bgp %s' % self.bgp_instance
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        return out

    def cli_add_command(self, command, undo=False):
        """add command to self.update_cmd and self.commands"""
        if undo and command.lower() not in ['quit', 'return']:
            cmd = 'undo ' + command
        else:
            cmd = command
        self.commands.append(cmd)
        if command.lower() not in ['quit', 'return']:
            self.updates_cmd.append(cmd)

    def cli_load_config(self, commands):
        """load config by cli"""
        if not self.module.check_mode:
            load_config(self.module, commands)

    def check_params(self):
        """Check all input params"""
        if not self.bgp_instance:
            self.module.fail_json(msg='Error: The bgp_instance can not be none.')
        if not self.peer_enable and (not self.advertise_router_type) and (not self.advertise_l2vpn_evpn):
            self.module.fail_json(msg='Error: The peer_enable, advertise_router_type, advertise_l2vpn_evpn can not be none at the same time.')
        if self.as_number:
            if not is_valid_as_number(self.as_number):
                self.module.fail_json(msg='Error: The parameter of as_number %s is invalid.' % self.as_number)
        if self.bgp_instance:
            if not is_valid_as_number(self.bgp_instance):
                self.module.fail_json(msg='Error: The parameter of bgp_instance %s is invalid.' % self.bgp_instance)
        if self.peer_address:
            if not is_valid_address(self.peer_address):
                self.module.fail_json(msg='Error: The %s is not a valid ip address.' % self.peer_address)
        if self.peer_group_name:
            if len(self.peer_group_name) > 47 or len(self.peer_group_name.replace(' ', '')) < 1:
                self.module.fail_json(msg='Error: peer group name is not in the range from 1 to 47.')
        if self.vpn_name:
            if len(self.vpn_name) > 31 or len(self.vpn_name.replace(' ', '')) < 1:
                self.module.fail_json(msg='Error: peer group name is not in the range from 1 to 31.')

    def get_proposed(self):
        """get proposed info"""
        if self.as_number:
            self.proposed['as_number'] = self.as_number
        if self.bgp_instance:
            self.proposed['bgp_instance'] = self.bgp_instance
        if self.peer_address:
            self.proposed['peer_address'] = self.peer_address
        if self.peer_group_name:
            self.proposed['peer_group_name'] = self.peer_group_name
        if self.peer_enable:
            self.proposed['peer_enable'] = self.peer_enable
        if self.advertise_router_type:
            self.proposed['advertise_router_type'] = self.advertise_router_type
        if self.vpn_name:
            self.proposed['vpn_name'] = self.vpn_name
        if self.advertise_l2vpn_evpn:
            self.proposed['advertise_l2vpn_evpn'] = self.advertise_l2vpn_evpn
        if not self.peer_enable or not self.advertise_l2vpn_evpn:
            if self.state:
                self.proposed['state'] = self.state

    def get_peers_enable(self):
        """get evpn peer address enable list"""
        if len(self.config_list) != 2:
            return None
        self.config_list = self.config.split('l2vpn-family evpn')
        get = re.findall('peer ([0-9]+.[0-9]+.[0-9]+.[0-9]+)\\s?as-number\\s?(\\S*)', self.config_list[0])
        if not get:
            return None
        else:
            peers = list()
            for item in get:
                cmd = 'peer %s enable' % item[0]
                exist = is_config_exist(self.config_list[1], cmd)
                if exist:
                    peers.append(dict(peer_address=item[0], as_number=item[1], peer_enable='true'))
                else:
                    peers.append(dict(peer_address=item[0], as_number=item[1], peer_enable='false'))
            return peers

    def get_peers_advertise_type(self):
        """get evpn peer address advertise type list"""
        if len(self.config_list) != 2:
            return None
        self.config_list = self.config.split('l2vpn-family evpn')
        get = re.findall('peer ([0-9]+.[0-9]+.[0-9]+.[0-9]+)\\s?as-number\\s?(\\S*)', self.config_list[0])
        if not get:
            return None
        else:
            peers = list()
            for item in get:
                cmd = 'peer %s advertise arp' % item[0]
                exist1 = is_config_exist(self.config_list[1], cmd)
                cmd = 'peer %s advertise irb' % item[0]
                exist2 = is_config_exist(self.config_list[1], cmd)
                if exist1:
                    peers.append(dict(peer_address=item[0], as_number=item[1], advertise_type='arp'))
                if exist2:
                    peers.append(dict(peer_address=item[0], as_number=item[1], advertise_type='irb'))
            return peers

    def get_peers_group_enable(self):
        """get evpn peer group name enable list"""
        if len(self.config_list) != 2:
            return None
        self.config_list = self.config.split('l2vpn-family evpn')
        get1 = re.findall('group (\\S+) external', self.config_list[0])
        get2 = re.findall('group (\\S+) internal', self.config_list[0])
        if not get1 and (not get2):
            return None
        else:
            peer_groups = list()
            for item in get1:
                cmd = 'peer %s enable' % item
                exist = is_config_exist(self.config_list[1], cmd)
                if exist:
                    peer_groups.append(dict(peer_group_name=item, peer_enable='true'))
                else:
                    peer_groups.append(dict(peer_group_name=item, peer_enable='false'))
            for item in get2:
                cmd = 'peer %s enable' % item
                exist = is_config_exist(self.config_list[1], cmd)
                if exist:
                    peer_groups.append(dict(peer_group_name=item, peer_enable='true'))
                else:
                    peer_groups.append(dict(peer_group_name=item, peer_enable='false'))
            return peer_groups

    def get_peer_groups_advertise_type(self):
        """get evpn peer group name advertise type list"""
        if len(self.config_list) != 2:
            return None
        self.config_list = self.config.split('l2vpn-family evpn')
        get1 = re.findall('group (\\S+) external', self.config_list[0])
        get2 = re.findall('group (\\S+) internal', self.config_list[0])
        if not get1 and (not get2):
            return None
        else:
            peer_groups = list()
            for item in get1:
                cmd = 'peer %s advertise arp' % item
                exist1 = is_config_exist(self.config_list[1], cmd)
                cmd = 'peer %s advertise irb' % item
                exist2 = is_config_exist(self.config_list[1], cmd)
                if exist1:
                    peer_groups.append(dict(peer_group_name=item, advertise_type='arp'))
                if exist2:
                    peer_groups.append(dict(peer_group_name=item, advertise_type='irb'))
            for item in get2:
                cmd = 'peer %s advertise arp' % item
                exist1 = is_config_exist(self.config_list[1], cmd)
                cmd = 'peer %s advertise irb' % item
                exist2 = is_config_exist(self.config_list[1], cmd)
                if exist1:
                    peer_groups.append(dict(peer_group_name=item, advertise_type='arp'))
                if exist2:
                    peer_groups.append(dict(peer_group_name=item, advertise_type='irb'))
            return peer_groups

    def get_existing(self):
        """get existing info"""
        if not self.config:
            return
        if self.bgp_instance:
            self.existing['bgp_instance'] = self.bgp_instance
        if self.peer_address and self.peer_enable:
            if self.l2vpn_evpn_exist:
                self.existing['peer_address_enable'] = self.get_peers_enable()
        if self.peer_group_name and self.peer_enable:
            if self.l2vpn_evpn_exist:
                self.existing['peer_group_enable'] = self.get_peers_group_enable()
        if self.peer_address and self.advertise_router_type:
            if self.l2vpn_evpn_exist:
                self.existing['peer_address_advertise_type'] = self.get_peers_advertise_type()
        if self.peer_group_name and self.advertise_router_type:
            if self.l2vpn_evpn_exist:
                self.existing['peer_group_advertise_type'] = self.get_peer_groups_advertise_type()
        if self.advertise_l2vpn_evpn and self.vpn_name:
            cmd = ' ipv4-family vpn-instance %s' % self.vpn_name
            exist = is_config_exist(self.config, cmd)
            if exist:
                self.existing['vpn_name'] = self.vpn_name
                l2vpn_cmd = 'advertise l2vpn evpn'
                l2vpn_exist = is_config_exist(self.config, l2vpn_cmd)
                if l2vpn_exist:
                    self.existing['advertise_l2vpn_evpn'] = 'enable'
                else:
                    self.existing['advertise_l2vpn_evpn'] = 'disable'

    def get_end_state(self):
        """get end state info"""
        self.config = self.get_current_config()
        if not self.config:
            return
        self.config_list = self.config.split('l2vpn-family evpn')
        if len(self.config_list) == 2:
            self.l2vpn_evpn_exist = True
        if self.bgp_instance:
            self.end_state['bgp_instance'] = self.bgp_instance
        if self.peer_address and self.peer_enable:
            if self.l2vpn_evpn_exist:
                self.end_state['peer_address_enable'] = self.get_peers_enable()
        if self.peer_group_name and self.peer_enable:
            if self.l2vpn_evpn_exist:
                self.end_state['peer_group_enable'] = self.get_peers_group_enable()
        if self.peer_address and self.advertise_router_type:
            if self.l2vpn_evpn_exist:
                self.end_state['peer_address_advertise_type'] = self.get_peers_advertise_type()
        if self.peer_group_name and self.advertise_router_type:
            if self.l2vpn_evpn_exist:
                self.end_state['peer_group_advertise_type'] = self.get_peer_groups_advertise_type()
        if self.advertise_l2vpn_evpn and self.vpn_name:
            cmd = ' ipv4-family vpn-instance %s' % self.vpn_name
            exist = is_config_exist(self.config, cmd)
            if exist:
                self.end_state['vpn_name'] = self.vpn_name
                l2vpn_cmd = 'advertise l2vpn evpn'
                l2vpn_exist = is_config_exist(self.config, l2vpn_cmd)
                if l2vpn_exist:
                    self.end_state['advertise_l2vpn_evpn'] = 'enable'
                else:
                    self.end_state['advertise_l2vpn_evpn'] = 'disable'

    def config_peer(self):
        """configure evpn bgp peer command"""
        if self.as_number and self.peer_address:
            cmd = 'peer %s as-number %s' % (self.peer_address, self.as_number)
            exist = is_config_exist(self.config, cmd)
            if not exist:
                self.module.fail_json(msg='Error:  The peer session %s does not exist or the peer already exists in another as-number.' % self.peer_address)
            cmd = 'bgp %s' % self.bgp_instance
            self.cli_add_command(cmd)
            cmd = 'l2vpn-family evpn'
            self.cli_add_command(cmd)
            exist_l2vpn = is_config_exist(self.config, cmd)
            if self.peer_enable:
                cmd = 'peer %s enable' % self.peer_address
                if exist_l2vpn:
                    exist = is_config_exist(self.config_list[1], cmd)
                    if self.peer_enable == 'true' and (not exist):
                        self.cli_add_command(cmd)
                        self.changed = True
                    elif self.peer_enable == 'false' and exist:
                        self.cli_add_command(cmd, undo=True)
                        self.changed = True
                else:
                    self.cli_add_command(cmd)
                    self.changed = True
            if self.advertise_router_type:
                cmd = 'peer %s advertise %s' % (self.peer_address, self.advertise_router_type)
                exist = is_config_exist(self.config, cmd)
                if self.state == 'present' and (not exist):
                    self.cli_add_command(cmd)
                    self.changed = True
                elif self.state == 'absent' and exist:
                    self.cli_add_command(cmd, undo=True)
                    self.changed = True
        elif self.peer_group_name:
            cmd_1 = 'group %s external' % self.peer_group_name
            exist_1 = is_config_exist(self.config, cmd_1)
            cmd_2 = 'group %s internal' % self.peer_group_name
            exist_2 = is_config_exist(self.config, cmd_2)
            exist = False
            if exist_1:
                exist = True
            if exist_2:
                exist = True
            if not exist:
                self.module.fail_json(msg='Error: The peer-group %s does not exist.' % self.peer_group_name)
            cmd = 'bgp %s' % self.bgp_instance
            self.cli_add_command(cmd)
            cmd = 'l2vpn-family evpn'
            self.cli_add_command(cmd)
            exist_l2vpn = is_config_exist(self.config, cmd)
            if self.peer_enable:
                cmd = 'peer %s enable' % self.peer_group_name
                if exist_l2vpn:
                    exist = is_config_exist(self.config_list[1], cmd)
                    if self.peer_enable == 'true' and (not exist):
                        self.cli_add_command(cmd)
                        self.changed = True
                    elif self.peer_enable == 'false' and exist:
                        self.cli_add_command(cmd, undo=True)
                        self.changed = True
                else:
                    self.cli_add_command(cmd)
                    self.changed = True
            if self.advertise_router_type:
                cmd = 'peer %s advertise %s' % (self.peer_group_name, self.advertise_router_type)
                exist = is_config_exist(self.config, cmd)
                if self.state == 'present' and (not exist):
                    self.cli_add_command(cmd)
                    self.changed = True
                elif self.state == 'absent' and exist:
                    self.cli_add_command(cmd, undo=True)
                    self.changed = True

    def config_advertise_l2vpn_evpn(self):
        """configure advertise l2vpn evpn"""
        cmd = 'ipv4-family vpn-instance %s' % self.vpn_name
        exist = is_config_exist(self.config, cmd)
        if not exist:
            self.module.fail_json(msg='Error: The VPN instance name %s does not exist.' % self.vpn_name)
        config_vpn_list = self.config.split(cmd)
        cmd = 'ipv4-family vpn-instance'
        exist_vpn = is_config_exist(config_vpn_list[1], cmd)
        cmd_l2vpn = 'advertise l2vpn evpn'
        if exist_vpn:
            config_vpn = config_vpn_list[1].split('ipv4-family vpn-instance')
            exist_l2vpn = is_config_exist(config_vpn[0], cmd_l2vpn)
        else:
            exist_l2vpn = is_config_exist(config_vpn_list[1], cmd_l2vpn)
        cmd = 'advertise l2vpn evpn'
        if self.advertise_l2vpn_evpn == 'enable' and (not exist_l2vpn):
            cmd = 'bgp %s' % self.bgp_instance
            self.cli_add_command(cmd)
            cmd = 'ipv4-family vpn-instance %s' % self.vpn_name
            self.cli_add_command(cmd)
            cmd = 'advertise l2vpn evpn'
            self.cli_add_command(cmd)
            self.changed = True
        elif self.advertise_l2vpn_evpn == 'disable' and exist_l2vpn:
            cmd = 'bgp %s' % self.bgp_instance
            self.cli_add_command(cmd)
            cmd = 'ipv4-family vpn-instance %s' % self.vpn_name
            self.cli_add_command(cmd)
            cmd = 'advertise l2vpn evpn'
            self.cli_add_command(cmd, undo=True)
            self.changed = True

    def work(self):
        """worker"""
        self.check_params()
        evpn_config = self.get_evpn_overlay_config()
        if not evpn_config:
            self.module.fail_json(msg='Error: evpn-overlay enable is not configured.')
        self.config = self.get_current_config()
        if not self.config:
            self.module.fail_json(msg='Error: Bgp instance %s does not exist.' % self.bgp_instance)
        self.config_list = self.config.split('l2vpn-family evpn')
        if len(self.config_list) == 2:
            self.l2vpn_evpn_exist = True
        self.get_existing()
        self.get_proposed()
        if self.peer_enable or self.advertise_router_type:
            self.config_peer()
        if self.advertise_l2vpn_evpn:
            self.config_advertise_l2vpn_evpn()
        if self.commands:
            self.cli_load_config(self.commands)
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)