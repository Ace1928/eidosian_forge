import logging
import os
import netaddr
from . import docker_base as base
def _create_config_zebra(self):
    c = base.CmdBuffer()
    c << 'hostname zebra'
    c << 'password zebra'
    c << 'log file {0}/zebra.log'.format(self.SHARED_VOLUME)
    c << 'debug zebra packet'
    c << 'debug zebra kernel'
    c << 'debug zebra rib'
    c << ''
    with open('{0}/zebra.conf'.format(self.config_dir), 'w') as f:
        LOG.info("[%s's new config]", self.name)
        LOG.info(str(c))
        f.writelines(str(c))