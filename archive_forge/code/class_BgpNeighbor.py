from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
class BgpNeighbor(object):
    """ Manages BGP peer configuration """

    def netconf_get_config(self, **kwargs):
        """ netconf_get_config """
        module = kwargs['module']
        conf_str = kwargs['conf_str']
        xml_str = get_nc_config(module, conf_str)
        return xml_str

    def netconf_set_config(self, **kwargs):
        """ netconf_set_config """
        module = kwargs['module']
        conf_str = kwargs['conf_str']
        xml_str = set_nc_config(module, conf_str)
        return xml_str

    def check_bgp_peer_args(self, **kwargs):
        """ check_bgp_peer_args """
        module = kwargs['module']
        result = dict()
        need_cfg = False
        vrf_name = module.params['vrf_name']
        if vrf_name:
            if len(vrf_name) > 31 or len(vrf_name) == 0:
                module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
        peer_addr = module.params['peer_addr']
        if peer_addr:
            if not check_ip_addr(ipaddr=peer_addr):
                module.fail_json(msg='Error: The peer_addr %s is invalid.' % peer_addr)
            need_cfg = True
        remote_as = module.params['remote_as']
        if remote_as:
            if len(remote_as) > 11 or len(remote_as) < 1:
                module.fail_json(msg='Error: The len of remote_as %s is out of [1 - 11].' % remote_as)
            need_cfg = True
        result['need_cfg'] = need_cfg
        return result

    def check_bgp_peer_other_args(self, **kwargs):
        """ check_bgp_peer_other_args """
        module = kwargs['module']
        result = dict()
        need_cfg = False
        peerip = module.params['peer_addr']
        vrf_name = module.params['vrf_name']
        if vrf_name:
            if len(vrf_name) > 31 or len(vrf_name) == 0:
                module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
        description = module.params['description']
        if description:
            if len(description) > 80 or len(description) < 1:
                module.fail_json(msg='Error: The len of description %s is out of [1 - 80].' % description)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<description></description>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<description>(.*)</description>.*', recv_xml)
                if re_find:
                    result['description'] = re_find
                    if re_find[0] != description:
                        need_cfg = True
                else:
                    need_cfg = True
        fake_as = module.params['fake_as']
        if fake_as:
            if len(fake_as) > 11 or len(fake_as) < 1:
                module.fail_json(msg='Error: The len of fake_as %s is out of [1 - 11].' % fake_as)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<fakeAs></fakeAs>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<fakeAs>(.*)</fakeAs>.*', recv_xml)
                if re_find:
                    result['fake_as'] = re_find
                    if re_find[0] != fake_as:
                        need_cfg = True
                else:
                    need_cfg = True
        dual_as = module.params['dual_as']
        if dual_as != 'no_use':
            if not fake_as:
                module.fail_json(msg='fake_as must exist.')
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<dualAs></dualAs>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<dualAs>(.*)</dualAs>.*', recv_xml)
                if re_find:
                    result['dual_as'] = re_find
                    if re_find[0] != dual_as:
                        need_cfg = True
                else:
                    need_cfg = True
        conventional = module.params['conventional']
        if conventional != 'no_use':
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<conventional></conventional>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<conventional>(.*)</conventional>.*', recv_xml)
                if re_find:
                    result['conventional'] = re_find
                    if re_find[0] != conventional:
                        need_cfg = True
                else:
                    need_cfg = True
        route_refresh = module.params['route_refresh']
        if route_refresh != 'no_use':
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<routeRefresh></routeRefresh>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<routeRefresh>(.*)</routeRefresh>.*', recv_xml)
                if re_find:
                    result['route_refresh'] = re_find
                    if re_find[0] != route_refresh:
                        need_cfg = True
                else:
                    need_cfg = True
        four_byte_as = module.params['four_byte_as']
        if four_byte_as != 'no_use':
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<fourByteAs></fourByteAs>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<fourByteAs>(.*)</fourByteAs>.*', recv_xml)
                if re_find:
                    result['four_byte_as'] = re_find
                    if re_find[0] != four_byte_as:
                        need_cfg = True
                else:
                    need_cfg = True
        is_ignore = module.params['is_ignore']
        if is_ignore != 'no_use':
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<isIgnore></isIgnore>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<isIgnore>(.*)</isIgnore>.*', recv_xml)
                if re_find:
                    result['is_ignore'] = re_find
                    if re_find[0] != is_ignore:
                        need_cfg = True
                else:
                    need_cfg = True
        local_if_name = module.params['local_if_name']
        if local_if_name:
            if len(local_if_name) > 63 or len(local_if_name) < 1:
                module.fail_json(msg='Error: The len of local_if_name %s is out of [1 - 63].' % local_if_name)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<localIfName></localIfName>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<localIfName>(.*)</localIfName>.*', recv_xml)
                if re_find:
                    result['local_if_name'] = re_find
                    if re_find[0].lower() != local_if_name.lower():
                        need_cfg = True
                else:
                    need_cfg = True
        ebgp_max_hop = module.params['ebgp_max_hop']
        if ebgp_max_hop:
            if int(ebgp_max_hop) > 255 or int(ebgp_max_hop) < 1:
                module.fail_json(msg='Error: The value of ebgp_max_hop %s is out of [1 - 255].' % ebgp_max_hop)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<ebgpMaxHop></ebgpMaxHop>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<ebgpMaxHop>(.*)</ebgpMaxHop>.*', recv_xml)
                if re_find:
                    result['ebgp_max_hop'] = re_find
                    if re_find[0] != ebgp_max_hop:
                        need_cfg = True
                else:
                    need_cfg = True
        valid_ttl_hops = module.params['valid_ttl_hops']
        if valid_ttl_hops:
            if int(valid_ttl_hops) > 255 or int(valid_ttl_hops) < 1:
                module.fail_json(msg='Error: The value of valid_ttl_hops %s is out of [1 - 255].' % valid_ttl_hops)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<validTtlHops></validTtlHops>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<validTtlHops>(.*)</validTtlHops>.*', recv_xml)
                if re_find:
                    result['valid_ttl_hops'] = re_find
                    if re_find[0] != valid_ttl_hops:
                        need_cfg = True
                else:
                    need_cfg = True
        connect_mode = module.params['connect_mode']
        if connect_mode:
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<connectMode></connectMode>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<connectMode>(.*)</connectMode>.*', recv_xml)
                if re_find:
                    result['connect_mode'] = re_find
                    if re_find[0] != connect_mode:
                        need_cfg = True
                else:
                    need_cfg = True
        is_log_change = module.params['is_log_change']
        if is_log_change != 'no_use':
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<isLogChange></isLogChange>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<isLogChange>(.*)</isLogChange>.*', recv_xml)
                if re_find:
                    result['is_log_change'] = re_find
                    if re_find[0] != is_log_change:
                        need_cfg = True
                else:
                    need_cfg = True
        pswd_type = module.params['pswd_type']
        if pswd_type:
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<pswdType></pswdType>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<pswdType>(.*)</pswdType>.*', recv_xml)
                if re_find:
                    result['pswd_type'] = re_find
                    if re_find[0] != pswd_type:
                        need_cfg = True
                else:
                    need_cfg = True
        pswd_cipher_text = module.params['pswd_cipher_text']
        if pswd_cipher_text:
            if len(pswd_cipher_text) > 255 or len(pswd_cipher_text) < 1:
                module.fail_json(msg='Error: The len of pswd_cipher_text %s is out of [1 - 255].' % pswd_cipher_text)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<pswdCipherText></pswdCipherText>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<pswdCipherText>(.*)</pswdCipherText>.*', recv_xml)
                if re_find:
                    result['pswd_cipher_text'] = re_find
                    if re_find[0] != pswd_cipher_text:
                        need_cfg = True
                else:
                    need_cfg = True
        keep_alive_time = module.params['keep_alive_time']
        if keep_alive_time:
            if int(keep_alive_time) > 21845 or len(keep_alive_time) < 0:
                module.fail_json(msg='Error: The len of keep_alive_time %s is out of [0 - 21845].' % keep_alive_time)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<keepAliveTime></keepAliveTime>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<keepAliveTime>(.*)</keepAliveTime>.*', recv_xml)
                if re_find:
                    result['keep_alive_time'] = re_find
                    if re_find[0] != keep_alive_time:
                        need_cfg = True
                else:
                    need_cfg = True
        hold_time = module.params['hold_time']
        if hold_time:
            if int(hold_time) != 0 and (int(hold_time) > 65535 or int(hold_time) < 3):
                module.fail_json(msg='Error: The value of hold_time %s is out of [0 or 3 - 65535].' % hold_time)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<holdTime></holdTime>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<holdTime>(.*)</holdTime>.*', recv_xml)
                if re_find:
                    result['hold_time'] = re_find
                    if re_find[0] != hold_time:
                        need_cfg = True
                else:
                    need_cfg = True
        min_hold_time = module.params['min_hold_time']
        if min_hold_time:
            if int(min_hold_time) != 0 and (int(min_hold_time) > 65535 or int(min_hold_time) < 20):
                module.fail_json(msg='Error: The value of min_hold_time %s is out of [0 or 20 - 65535].' % min_hold_time)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<minHoldTime></minHoldTime>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<minHoldTime>(.*)</minHoldTime>.*', recv_xml)
                if re_find:
                    result['min_hold_time'] = re_find
                    if re_find[0] != min_hold_time:
                        need_cfg = True
                else:
                    need_cfg = True
        key_chain_name = module.params['key_chain_name']
        if key_chain_name:
            if len(key_chain_name) > 47 or len(key_chain_name) < 1:
                module.fail_json(msg='Error: The len of key_chain_name %s is out of [1 - 47].' % key_chain_name)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<keyChainName></keyChainName>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<keyChainName>(.*)</keyChainName>.*', recv_xml)
                if re_find:
                    result['key_chain_name'] = re_find
                    if re_find[0] != key_chain_name:
                        need_cfg = True
                else:
                    need_cfg = True
        conn_retry_time = module.params['conn_retry_time']
        if conn_retry_time:
            if int(conn_retry_time) > 65535 or int(conn_retry_time) < 1:
                module.fail_json(msg='Error: The value of conn_retry_time %s is out of [1 - 65535].' % conn_retry_time)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<connRetryTime></connRetryTime>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<connRetryTime>(.*)</connRetryTime>.*', recv_xml)
                if re_find:
                    result['conn_retry_time'] = re_find
                    if re_find[0] != conn_retry_time:
                        need_cfg = True
                else:
                    need_cfg = True
        tcp_mss = module.params['tcp_MSS']
        if tcp_mss:
            if int(tcp_mss) > 4096 or int(tcp_mss) < 176:
                module.fail_json(msg='Error: The value of tcp_mss %s is out of [176 - 4096].' % tcp_mss)
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<tcpMSS></tcpMSS>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<tcpMSS>(.*)</tcpMSS>.*', recv_xml)
                if re_find:
                    result['tcp_MSS'] = re_find
                    if re_find[0] != tcp_mss:
                        need_cfg = True
                else:
                    need_cfg = True
        mpls_local_ifnet_disable = module.params['mpls_local_ifnet_disable']
        if mpls_local_ifnet_disable != 'no_use':
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<mplsLocalIfnetDisable></mplsLocalIfnetDisable>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<mplsLocalIfnetDisable>(.*)</mplsLocalIfnetDisable>.*', recv_xml)
                if re_find:
                    result['mpls_local_ifnet_disable'] = re_find
                    if re_find[0] != mpls_local_ifnet_disable:
                        need_cfg = True
                else:
                    need_cfg = True
        prepend_global_as = module.params['prepend_global_as']
        if prepend_global_as != 'no_use':
            if not fake_as:
                module.fail_json(msg='fake_as must exist.')
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<prependGlobalAs></prependGlobalAs>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<prependGlobalAs>(.*)</prependGlobalAs>.*', recv_xml)
                if re_find:
                    result['prepend_global_as'] = re_find
                    if re_find[0] != prepend_global_as:
                        need_cfg = True
                else:
                    need_cfg = True
        prepend_fake_as = module.params['prepend_fake_as']
        if prepend_fake_as != 'no_use':
            if not fake_as:
                module.fail_json(msg='fake_as must exist.')
            conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<prependFakeAs></prependFakeAs>' + CE_GET_BGP_PEER_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<prependFakeAs>(.*)</prependFakeAs>.*', recv_xml)
                if re_find:
                    result['prepend_fake_as'] = re_find
                    if re_find[0] != prepend_fake_as:
                        need_cfg = True
                else:
                    need_cfg = True
        result['need_cfg'] = need_cfg
        return result

    def check_peer_bfd_merge_args(self, **kwargs):
        """ check_peer_bfd_merge_args """
        module = kwargs['module']
        result = dict()
        need_cfg = False
        state = module.params['state']
        if state == 'absent':
            result['need_cfg'] = need_cfg
            return result
        vrf_name = module.params['vrf_name']
        if vrf_name:
            if len(vrf_name) > 31 or len(vrf_name) == 0:
                module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
        peer_addr = module.params['peer_addr']
        is_bfd_block = module.params['is_bfd_block']
        if is_bfd_block != 'no_use':
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<isBfdBlock></isBfdBlock>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<isBfdBlock>(.*)</isBfdBlock>.*', recv_xml)
                if re_find:
                    result['is_bfd_block'] = re_find
                    if re_find[0] != is_bfd_block:
                        need_cfg = True
                else:
                    need_cfg = True
        multiplier = module.params['multiplier']
        if multiplier:
            if int(multiplier) > 50 or int(multiplier) < 3:
                module.fail_json(msg='Error: The value of multiplier %s is out of [3 - 50].' % multiplier)
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<multiplier></multiplier>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<multiplier>(.*)</multiplier>.*', recv_xml)
                if re_find:
                    result['multiplier'] = re_find
                    if re_find[0] != multiplier:
                        need_cfg = True
                else:
                    need_cfg = True
        is_bfd_enable = module.params['is_bfd_enable']
        if is_bfd_enable != 'no_use':
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<isBfdEnable></isBfdEnable>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<isBfdEnable>(.*)</isBfdEnable>.*', recv_xml)
                if re_find:
                    result['is_bfd_enable'] = re_find
                    if re_find[0] != is_bfd_enable:
                        need_cfg = True
                else:
                    need_cfg = True
        rx_interval = module.params['rx_interval']
        if rx_interval:
            if int(rx_interval) > 1000 or int(rx_interval) < 50:
                module.fail_json(msg='Error: The value of rx_interval %s is out of [50 - 1000].' % rx_interval)
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<rxInterval></rxInterval>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<rxInterval>(.*)</rxInterval>.*', recv_xml)
                if re_find:
                    result['rx_interval'] = re_find
                    if re_find[0] != rx_interval:
                        need_cfg = True
                else:
                    need_cfg = True
        tx_interval = module.params['tx_interval']
        if tx_interval:
            if int(tx_interval) > 1000 or int(tx_interval) < 50:
                module.fail_json(msg='Error: The value of tx_interval %s is out of [50 - 1000].' % tx_interval)
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<txInterval></txInterval>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<txInterval>(.*)</txInterval>.*', recv_xml)
                if re_find:
                    result['tx_interval'] = re_find
                    if re_find[0] != tx_interval:
                        need_cfg = True
                else:
                    need_cfg = True
        is_single_hop = module.params['is_single_hop']
        if is_single_hop != 'no_use':
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<isSingleHop></isSingleHop>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<isSingleHop>(.*)</isSingleHop>.*', recv_xml)
                if re_find:
                    result['is_single_hop'] = re_find
                    if re_find[0] != is_single_hop:
                        need_cfg = True
                else:
                    need_cfg = True
        result['need_cfg'] = need_cfg
        return result

    def check_peer_bfd_delete_args(self, **kwargs):
        """ check_peer_bfd_delete_args """
        module = kwargs['module']
        result = dict()
        need_cfg = False
        state = module.params['state']
        if state == 'present':
            result['need_cfg'] = need_cfg
            return result
        vrf_name = module.params['vrf_name']
        if vrf_name:
            if len(vrf_name) > 31 or len(vrf_name) == 0:
                module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
        peer_addr = module.params['peer_addr']
        is_bfd_block = module.params['is_bfd_block']
        if is_bfd_block != 'no_use':
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<isBfdBlock></isBfdBlock>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<isBfdBlock>(.*)</isBfdBlock>.*', recv_xml)
                if re_find:
                    result['is_bfd_block'] = re_find
                    if re_find[0] == is_bfd_block:
                        need_cfg = True
        multiplier = module.params['multiplier']
        if multiplier:
            if int(multiplier) > 50 or int(multiplier) < 3:
                module.fail_json(msg='Error: The value of multiplier %s is out of [3 - 50].' % multiplier)
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<multiplier></multiplier>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<multiplier>(.*)</multiplier>.*', recv_xml)
                if re_find:
                    result['multiplier'] = re_find
                    if re_find[0] == multiplier:
                        need_cfg = True
        is_bfd_enable = module.params['is_bfd_enable']
        if is_bfd_enable != 'no_use':
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<isBfdEnable></isBfdEnable>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<isBfdEnable>(.*)</isBfdEnable>.*', recv_xml)
                if re_find:
                    result['is_bfd_enable'] = re_find
                    if re_find[0] == is_bfd_enable:
                        need_cfg = True
        rx_interval = module.params['rx_interval']
        if rx_interval:
            if int(rx_interval) > 1000 or int(rx_interval) < 50:
                module.fail_json(msg='Error: The value of rx_interval %s is out of [50 - 1000].' % rx_interval)
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<rxInterval></rxInterval>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<rxInterval>(.*)</rxInterval>.*', recv_xml)
                if re_find:
                    result['rx_interval'] = re_find
                    if re_find[0] == rx_interval:
                        need_cfg = True
        tx_interval = module.params['tx_interval']
        if tx_interval:
            if int(tx_interval) > 1000 or int(tx_interval) < 50:
                module.fail_json(msg='Error: The value of tx_interval %s is out of [50 - 1000].' % tx_interval)
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<txInterval></txInterval>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<txInterval>(.*)</txInterval>.*', recv_xml)
                if re_find:
                    result['tx_interval'] = re_find
                    if re_find[0] == tx_interval:
                        need_cfg = True
        is_single_hop = module.params['is_single_hop']
        if is_single_hop != 'no_use':
            conf_str = CE_GET_PEER_BFD_HEADER % (vrf_name, peer_addr) + '<isSingleHop></isSingleHop>' + CE_GET_PEER_BFD_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                pass
            else:
                re_find = re.findall('.*<isSingleHop>(.*)</isSingleHop>.*', recv_xml)
                if re_find:
                    result['is_single_hop'] = re_find
                    if re_find[0] == is_single_hop:
                        need_cfg = True
        result['need_cfg'] = need_cfg
        return result

    def get_bgp_peer(self, **kwargs):
        """ get_bgp_peer """
        module = kwargs['module']
        peerip = module.params['peer_addr']
        vrf_name = module.params['vrf_name']
        if vrf_name:
            if len(vrf_name) > 31 or len(vrf_name) == 0:
                module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
        conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + '<remoteAs></remoteAs>' + CE_GET_BGP_PEER_TAIL
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<peerAddr>(.*)</peerAddr>.*\\s.*<remoteAs>(.*)</remoteAs>.*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def get_bgp_del_peer(self, **kwargs):
        """ get_bgp_del_peer """
        module = kwargs['module']
        peerip = module.params['peer_addr']
        vrf_name = module.params['vrf_name']
        if vrf_name:
            if len(vrf_name) > 31 or len(vrf_name) == 0:
                module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
        conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + CE_GET_BGP_PEER_TAIL
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<peerAddr>(.*)</peerAddr>.*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def merge_bgp_peer(self, **kwargs):
        """ merge_bgp_peer """
        module = kwargs['module']
        vrf_name = module.params['vrf_name']
        peer_addr = module.params['peer_addr']
        remote_as = module.params['remote_as']
        conf_str = CE_MERGE_BGP_PEER_HEADER % (vrf_name, peer_addr) + '<remoteAs>%s</remoteAs>' % remote_as + CE_MERGE_BGP_PEER_TAIL
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge bgp peer failed.')
        cmds = []
        cmd = 'peer %s as-number %s' % (peer_addr, remote_as)
        cmds.append(cmd)
        return cmds

    def create_bgp_peer(self, **kwargs):
        """ create_bgp_peer """
        module = kwargs['module']
        vrf_name = module.params['vrf_name']
        peer_addr = module.params['peer_addr']
        remote_as = module.params['remote_as']
        conf_str = CE_CREATE_BGP_PEER_HEADER % (vrf_name, peer_addr) + '<remoteAs>%s</remoteAs>' % remote_as + CE_CREATE_BGP_PEER_TAIL
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Create bgp peer failed.')
        cmds = []
        cmd = 'peer %s as-number %s' % (peer_addr, remote_as)
        cmds.append(cmd)
        return cmds

    def delete_bgp_peer(self, **kwargs):
        """ delete_bgp_peer """
        module = kwargs['module']
        vrf_name = module.params['vrf_name']
        peer_addr = module.params['peer_addr']
        conf_str = CE_DELETE_BGP_PEER_HEADER % (vrf_name, peer_addr) + CE_DELETE_BGP_PEER_TAIL
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Delete bgp peer failed.')
        cmds = []
        cmd = 'undo peer %s' % peer_addr
        cmds.append(cmd)
        return cmds

    def merge_bgp_peer_other(self, **kwargs):
        """ merge_bgp_peer """
        module = kwargs['module']
        vrf_name = module.params['vrf_name']
        peer_addr = module.params['peer_addr']
        conf_str = CE_MERGE_BGP_PEER_HEADER % (vrf_name, peer_addr)
        cmds = []
        description = module.params['description']
        if description:
            conf_str += '<description>%s</description>' % description
            cmd = 'peer %s description %s' % (peer_addr, description)
            cmds.append(cmd)
        fake_as = module.params['fake_as']
        if fake_as:
            conf_str += '<fakeAs>%s</fakeAs>' % fake_as
            cmd = 'peer %s local-as %s' % (peer_addr, fake_as)
            cmds.append(cmd)
        dual_as = module.params['dual_as']
        if dual_as != 'no_use':
            conf_str += '<dualAs>%s</dualAs>' % dual_as
            if dual_as == 'true':
                cmd = 'peer %s local-as %s dual-as' % (peer_addr, fake_as)
            else:
                cmd = 'peer %s local-as %s' % (peer_addr, fake_as)
            cmds.append(cmd)
        conventional = module.params['conventional']
        if conventional != 'no_use':
            conf_str += '<conventional>%s</conventional>' % conventional
            if conventional == 'true':
                cmd = 'peer %s capability-advertise conventional' % peer_addr
            else:
                cmd = 'undo peer %s capability-advertise conventional' % peer_addr
            cmds.append(cmd)
        route_refresh = module.params['route_refresh']
        if route_refresh != 'no_use':
            conf_str += '<routeRefresh>%s</routeRefresh>' % route_refresh
            if route_refresh == 'true':
                cmd = 'peer %s capability-advertise route-refresh' % peer_addr
            else:
                cmd = 'undo peer %s capability-advertise route-refresh' % peer_addr
            cmds.append(cmd)
        four_byte_as = module.params['four_byte_as']
        if four_byte_as != 'no_use':
            conf_str += '<fourByteAs>%s</fourByteAs>' % four_byte_as
            if four_byte_as == 'true':
                cmd = 'peer %s capability-advertise 4-byte-as' % peer_addr
            else:
                cmd = 'undo peer %s capability-advertise 4-byte-as' % peer_addr
            cmds.append(cmd)
        is_ignore = module.params['is_ignore']
        if is_ignore != 'no_use':
            conf_str += '<isIgnore>%s</isIgnore>' % is_ignore
            if is_ignore == 'true':
                cmd = 'peer %s ignore' % peer_addr
            else:
                cmd = 'undo peer %s ignore' % peer_addr
            cmds.append(cmd)
        local_if_name = module.params['local_if_name']
        if local_if_name:
            conf_str += '<localIfName>%s</localIfName>' % local_if_name
            cmd = 'peer %s connect-interface %s' % (peer_addr, local_if_name)
            cmds.append(cmd)
        ebgp_max_hop = module.params['ebgp_max_hop']
        if ebgp_max_hop:
            conf_str += '<ebgpMaxHop>%s</ebgpMaxHop>' % ebgp_max_hop
            cmd = 'peer %s ebgp-max-hop %s' % (peer_addr, ebgp_max_hop)
            cmds.append(cmd)
        valid_ttl_hops = module.params['valid_ttl_hops']
        if valid_ttl_hops:
            conf_str += '<validTtlHops>%s</validTtlHops>' % valid_ttl_hops
            cmd = 'peer %s valid-ttl-hops %s' % (peer_addr, valid_ttl_hops)
            cmds.append(cmd)
        connect_mode = module.params['connect_mode']
        if connect_mode:
            if connect_mode == 'listenOnly':
                cmd = 'peer %s listen-only' % peer_addr
                cmds.append(cmd)
            elif connect_mode == 'connectOnly':
                cmd = 'peer %s connect-only' % peer_addr
                cmds.append(cmd)
            elif connect_mode == 'both':
                connect_mode = 'null'
                cmd = 'peer %s listen-only' % peer_addr
                cmds.append(cmd)
                cmd = 'peer %s connect-only' % peer_addr
                cmds.append(cmd)
            conf_str += '<connectMode>%s</connectMode>' % connect_mode
        is_log_change = module.params['is_log_change']
        if is_log_change != 'no_use':
            conf_str += '<isLogChange>%s</isLogChange>' % is_log_change
            if is_log_change == 'true':
                cmd = 'peer %s log-change' % peer_addr
            else:
                cmd = 'undo peer %s log-change' % peer_addr
            cmds.append(cmd)
        pswd_type = module.params['pswd_type']
        if pswd_type:
            conf_str += '<pswdType>%s</pswdType>' % pswd_type
        pswd_cipher_text = module.params['pswd_cipher_text']
        if pswd_cipher_text:
            conf_str += '<pswdCipherText>%s</pswdCipherText>' % pswd_cipher_text
            if pswd_type == 'cipher':
                cmd = 'peer %s password cipher %s' % (peer_addr, pswd_cipher_text)
            elif pswd_type == 'simple':
                cmd = 'peer %s password simple %s' % (peer_addr, pswd_cipher_text)
            cmds.append(cmd)
        keep_alive_time = module.params['keep_alive_time']
        if keep_alive_time:
            conf_str += '<keepAliveTime>%s</keepAliveTime>' % keep_alive_time
            cmd = 'peer %s timer keepalive %s' % (peer_addr, keep_alive_time)
            cmds.append(cmd)
        hold_time = module.params['hold_time']
        if hold_time:
            conf_str += '<holdTime>%s</holdTime>' % hold_time
            cmd = 'peer %s timer hold %s' % (peer_addr, hold_time)
            cmds.append(cmd)
        min_hold_time = module.params['min_hold_time']
        if min_hold_time:
            conf_str += '<minHoldTime>%s</minHoldTime>' % min_hold_time
            cmd = 'peer %s timer min-holdtime %s' % (peer_addr, min_hold_time)
            cmds.append(cmd)
        key_chain_name = module.params['key_chain_name']
        if key_chain_name:
            conf_str += '<keyChainName>%s</keyChainName>' % key_chain_name
            cmd = 'peer %s keychain %s' % (peer_addr, key_chain_name)
            cmds.append(cmd)
        conn_retry_time = module.params['conn_retry_time']
        if conn_retry_time:
            conf_str += '<connRetryTime>%s</connRetryTime>' % conn_retry_time
            cmd = 'peer %s timer connect-retry %s' % (peer_addr, conn_retry_time)
            cmds.append(cmd)
        tcp_mss = module.params['tcp_MSS']
        if tcp_mss:
            conf_str += '<tcpMSS>%s</tcpMSS>' % tcp_mss
            cmd = 'peer %s tcp-mss %s' % (peer_addr, tcp_mss)
            cmds.append(cmd)
        mpls_local_ifnet_disable = module.params['mpls_local_ifnet_disable']
        if mpls_local_ifnet_disable != 'no_use':
            conf_str += '<mplsLocalIfnetDisable>%s</mplsLocalIfnetDisable>' % mpls_local_ifnet_disable
            if mpls_local_ifnet_disable == 'false':
                cmd = 'undo peer %s mpls-local-ifnet disable' % peer_addr
            else:
                cmd = 'peer %s mpls-local-ifnet disable' % peer_addr
            cmds.append(cmd)
        prepend_global_as = module.params['prepend_global_as']
        if prepend_global_as != 'no_use':
            conf_str += '<prependGlobalAs>%s</prependGlobalAs>' % prepend_global_as
            if prepend_global_as == 'true':
                cmd = 'peer %s local-as %s prepend-global-as' % (peer_addr, fake_as)
            else:
                cmd = 'undo peer %s local-as %s prepend-global-as' % (peer_addr, fake_as)
            cmds.append(cmd)
        prepend_fake_as = module.params['prepend_fake_as']
        if prepend_fake_as != 'no_use':
            conf_str += '<prependFakeAs>%s</prependFakeAs>' % prepend_fake_as
            if prepend_fake_as == 'true':
                cmd = 'peer %s local-as %s prepend-local-as' % (peer_addr, fake_as)
            else:
                cmd = 'undo peer %s local-as %s prepend-local-as' % (peer_addr, fake_as)
            cmds.append(cmd)
        conf_str += CE_MERGE_BGP_PEER_TAIL
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge bgp peer other failed.')
        return cmds

    def merge_peer_bfd(self, **kwargs):
        """ merge_peer_bfd """
        module = kwargs['module']
        vrf_name = module.params['vrf_name']
        peer_addr = module.params['peer_addr']
        conf_str = CE_MERGE_PEER_BFD_HEADER % (vrf_name, peer_addr)
        cmds = []
        is_bfd_block = module.params['is_bfd_block']
        if is_bfd_block != 'no_use':
            conf_str += '<isBfdBlock>%s</isBfdBlock>' % is_bfd_block
            if is_bfd_block == 'true':
                cmd = 'peer %s bfd block' % peer_addr
            else:
                cmd = 'undo peer %s bfd block' % peer_addr
            cmds.append(cmd)
        multiplier = module.params['multiplier']
        if multiplier:
            conf_str += '<multiplier>%s</multiplier>' % multiplier
            cmd = 'peer %s bfd detect-multiplier %s' % (peer_addr, multiplier)
            cmds.append(cmd)
        is_bfd_enable = module.params['is_bfd_enable']
        if is_bfd_enable != 'no_use':
            conf_str += '<isBfdEnable>%s</isBfdEnable>' % is_bfd_enable
            if is_bfd_enable == 'true':
                cmd = 'peer %s bfd enable' % peer_addr
            else:
                cmd = 'undo peer %s bfd enable' % peer_addr
            cmds.append(cmd)
        rx_interval = module.params['rx_interval']
        if rx_interval:
            conf_str += '<rxInterval>%s</rxInterval>' % rx_interval
            cmd = 'peer %s bfd min-rx-interval %s' % (peer_addr, rx_interval)
            cmds.append(cmd)
        tx_interval = module.params['tx_interval']
        if tx_interval:
            conf_str += '<txInterval>%s</txInterval>' % tx_interval
            cmd = 'peer %s bfd min-tx-interval %s' % (peer_addr, tx_interval)
            cmds.append(cmd)
        is_single_hop = module.params['is_single_hop']
        if is_single_hop != 'no_use':
            conf_str += '<isSingleHop>%s</isSingleHop>' % is_single_hop
            if is_single_hop == 'true':
                cmd = 'peer %s bfd enable single-hop-prefer' % peer_addr
            else:
                cmd = 'undo peer %s bfd enable single-hop-prefer' % peer_addr
            cmds.append(cmd)
        conf_str += CE_MERGE_PEER_BFD_TAIL
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge peer bfd failed.')
        return cmds

    def delete_peer_bfd(self, **kwargs):
        """ delete_peer_bfd """
        module = kwargs['module']
        vrf_name = module.params['vrf_name']
        peer_addr = module.params['peer_addr']
        conf_str = CE_DELETE_PEER_BFD_HEADER % (vrf_name, peer_addr)
        cmds = []
        is_bfd_block = module.params['is_bfd_block']
        if is_bfd_block != 'no_use':
            conf_str += '<isBfdBlock>%s</isBfdBlock>' % is_bfd_block
            cmd = 'undo peer %s bfd block' % peer_addr
            cmds.append(cmd)
        multiplier = module.params['multiplier']
        if multiplier:
            conf_str += '<multiplier>%s</multiplier>' % multiplier
            cmd = 'undo peer %s bfd detect-multiplier %s' % (peer_addr, multiplier)
            cmds.append(cmd)
        is_bfd_enable = module.params['is_bfd_enable']
        if is_bfd_enable != 'no_use':
            conf_str += '<isBfdEnable>%s</isBfdEnable>' % is_bfd_enable
            cmd = 'undo peer %s bfd enable' % peer_addr
            cmds.append(cmd)
        rx_interval = module.params['rx_interval']
        if rx_interval:
            conf_str += '<rxInterval>%s</rxInterval>' % rx_interval
            cmd = 'undo peer %s bfd min-rx-interval %s' % (peer_addr, rx_interval)
            cmds.append(cmd)
        tx_interval = module.params['tx_interval']
        if tx_interval:
            conf_str += '<txInterval>%s</txInterval>' % tx_interval
            cmd = 'undo peer %s bfd min-tx-interval %s' % (peer_addr, tx_interval)
            cmds.append(cmd)
        is_single_hop = module.params['is_single_hop']
        if is_single_hop != 'no_use':
            conf_str += '<isSingleHop>%s</isSingleHop>' % is_single_hop
            cmd = 'undo peer %s bfd enable single-hop-prefer' % peer_addr
            cmds.append(cmd)
        conf_str += CE_DELETE_PEER_BFD_TAIL
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Delete peer bfd failed.')
        return cmds