import logging
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from os_ken.lib import netdevice
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from . import base
def _remove_inet_addr(intf_inet, addr):
    addr_list = intf_inet.split(',')
    if addr not in addr_list:
        LOG.debug('Interface "%s" does not have "ifaddr": %s', intf.ifname, addr)
        return intf_inet
    else:
        addr_list.remove(addr)
        return ','.join(addr_list)