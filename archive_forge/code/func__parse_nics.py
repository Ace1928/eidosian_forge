import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
def _parse_nics(cs, args):
    supports_auto_alloc = cs.api_version >= api_versions.APIVersion('2.37')
    supports_nic_tags = _supports_nic_tags(cs)
    nic_keys = {'net-id', 'v4-fixed-ip', 'v6-fixed-ip', 'port-id', 'net-name'}
    if supports_auto_alloc and supports_nic_tags:
        nic_keys.add('tag')
        err_msg = _("Invalid nic argument '%s'. Nic arguments must be of the form --nic <auto,none,net-id=net-uuid,net-name=network-name,v4-fixed-ip=ip-addr,v6-fixed-ip=ip-addr,port-id=port-uuid,tag=tag>, with only one of net-id, net-name or port-id specified. Specifying a --nic of auto or none cannot be used with any other --nic value.")
    elif supports_auto_alloc and (not supports_nic_tags):
        err_msg = _("Invalid nic argument '%s'. Nic arguments must be of the form --nic <auto,none,net-id=net-uuid,net-name=network-name,v4-fixed-ip=ip-addr,v6-fixed-ip=ip-addr,port-id=port-uuid>, with only one of net-id, net-name or port-id specified. Specifying a --nic of auto or none cannot be used with any other --nic value.")
    elif not supports_auto_alloc and supports_nic_tags:
        nic_keys.add('tag')
        err_msg = _("Invalid nic argument '%s'. Nic arguments must be of the form --nic <net-id=net-uuid,net-name=network-name,v4-fixed-ip=ip-addr,v6-fixed-ip=ip-addr,port-id=port-uuid,tag=tag>, with only one of net-id, net-name or port-id specified.")
    else:
        err_msg = _("Invalid nic argument '%s'. Nic arguments must be of the form --nic <net-id=net-uuid,net-name=network-name,v4-fixed-ip=ip-addr,v6-fixed-ip=ip-addr,port-id=port-uuid>, with only one of net-id, net-name or port-id specified.")
    auto_or_none = False
    nics = []
    for nic_str in args.nics:
        nic_info = {}
        nic_info_set = False
        for kv_str in nic_str.split(','):
            if auto_or_none:
                raise exceptions.CommandError(_("'auto' or 'none' cannot be used with any other nic arguments"))
            try:
                if kv_str in ('auto', 'none'):
                    if not supports_auto_alloc:
                        raise exceptions.CommandError(err_msg % nic_str)
                    if nic_info_set:
                        raise exceptions.CommandError(_("'auto' or 'none' cannot be used with any other nic arguments"))
                    nics.append(kv_str)
                    auto_or_none = True
                    continue
                k, v = kv_str.split('=', 1)
            except ValueError:
                raise exceptions.CommandError(err_msg % nic_str)
            if k in nic_keys:
                if k == 'net-name':
                    k = 'net-id'
                    v = _find_network_id(cs, v)
                if k in nic_info:
                    raise exceptions.CommandError(err_msg % nic_str)
                nic_info[k] = v
                nic_info_set = True
            else:
                raise exceptions.CommandError(err_msg % nic_str)
        if auto_or_none:
            continue
        if 'v4-fixed-ip' in nic_info and (not netutils.is_valid_ipv4(nic_info['v4-fixed-ip'])):
            raise exceptions.CommandError(_('Invalid ipv4 address.'))
        if 'v6-fixed-ip' in nic_info and (not netutils.is_valid_ipv6(nic_info['v6-fixed-ip'])):
            raise exceptions.CommandError(_('Invalid ipv6 address.'))
        if bool(nic_info.get('net-id')) == bool(nic_info.get('port-id')):
            raise exceptions.CommandError(err_msg % nic_str)
        nics.append(nic_info)
    if nics:
        if auto_or_none:
            if len(nics) > 1:
                raise exceptions.CommandError(err_msg % nic_str)
            nics = nics[0]
    elif supports_auto_alloc:
        nics = 'auto'
    return nics