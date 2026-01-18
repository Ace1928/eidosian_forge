import logging
import os
import netaddr
from . import docker_base as base
def _create_config_debian(self):
    c = base.CmdBuffer()
    c << 'vtysh_enable=yes'
    c << 'zebra_options="  --daemon -A 127.0.0.1"'
    c << 'bgpd_options="   --daemon -A 127.0.0.1"'
    c << 'ospfd_options="  --daemon -A 127.0.0.1"'
    c << 'ospf6d_options=" --daemon -A ::1"'
    c << 'ripd_options="   --daemon -A 127.0.0.1"'
    c << 'ripngd_options=" --daemon -A ::1"'
    c << 'isisd_options="  --daemon -A 127.0.0.1"'
    c << 'babeld_options=" --daemon -A 127.0.0.1"'
    c << 'watchquagga_enable=yes'
    c << 'watchquagga_options=(--daemon)'
    with open('{0}/debian.conf'.format(self.config_dir), 'w') as f:
        LOG.info("[%s's new config]", self.name)
        LOG.info(str(c))
        f.writelines(str(c))